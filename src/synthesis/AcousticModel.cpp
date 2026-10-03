#include "AcousticModel.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <new>
#include <random>
#include <utility>

namespace sv::synthesis
{
namespace
{
constexpr std::size_t maxFrames = 64 * 1024 * 1024 / 388;

juce::Result fail(const juce::String& message)
{
    return juce::Result::fail("Acoustic model: " + message);
}

std::uint32_t readWord(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset]) | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

juce::Result checkShape(const DnniTensor& tensor, std::size_t frames, std::size_t channels, const char* name)
{
    if (tensor.frames != frames || tensor.channels != channels || tensor.values.size() != frames * channels)
    {
        return fail(juce::String(name) + " returned an unexpected tensor shape");
    }
    if (!std::all_of(tensor.values.begin(), tensor.values.end(), [](float value)
                     { return std::isfinite(value); }))
    {
        return fail(juce::String(name) + " returned a non-finite value");
    }
    return juce::Result::ok();
}

class GaussianNoise
{
public:
    explicit GaussianNoise(std::uint32_t seed) : engine(seed) {}

    float next()
    {
        if (hasSpare)
        {
            hasSpare = false;
            return spare;
        }
        for (;;)
        {
            float first = static_cast<float>(engine()) * 0x1p-31f - 1.0f;
            float second = static_cast<float>(engine()) * 0x1p-31f - 1.0f;
            float squared = first * first + second * second;
            if (squared >= 1.0f || first == 0.0f || second == 0.0f)
            {
                continue;
            }
            float logarithm = 0.0f;
            if (squared <= 0.0001f)
            {
                const int exponent = std::ilogb(std::max(std::abs(first), std::abs(second)));
                first = std::scalbn(first, -exponent);
                second = std::scalbn(second, -exponent);
                squared = first * first + second * second;
                logarithm = std::log(squared) + static_cast<float>(exponent) * 1.3862943649291992f;
            }
            else
            {
                logarithm = std::log(squared);
            }
            const float scale = std::sqrt(-2.0f * logarithm / squared);
            spare = second * scale;
            hasSpare = true;
            return first * scale;
        }
    }

private:
    std::mt19937 engine;
    float spare = 0.0f;
    bool hasSpare = false;
};
} // namespace

std::size_t AcousticModel::State::getBytes() const noexcept
{
    return phonemeEncoder.getBytes() + contextProjection.getBytes() + contextNetwork.getBytes() + latentProjection.getBytes() + latentNetwork.getBytes() + latentHead.getBytes() + spectralNetwork.getBytes() + spectralHead.getBytes() + auxiliaryNetwork.getBytes() + auxiliaryHead.getBytes();
}

juce::Result AcousticModel::load(const DnniReader& reader, std::size_t rootNode)
try
{
    const auto& nodes = reader.getNodes();
    if (rootNode >= nodes.size() || nodes[rootNode].type != "_rldtg0" || nodes[rootNode].children.size() != 2)
    {
        return fail("expected an _rldtg0 root with model and style children");
    }
    const auto& model = nodes[nodes[rootNode].children[0]];
    const auto& style = nodes[nodes[rootNode].children[1]];
    if (model.type != "_rldms0" || model.children.size() != 8 || style.type != "_stbkv1" || style.children.size() != 2)
    {
        return fail("unsupported acoustic model or style layout");
    }
    const auto& context = nodes[model.children[2]];
    const auto& latent = nodes[model.children[3]];
    if (context.type != "_vqctx1" || context.children.size() != 4 || latent.type != "_didsv0" || latent.children.size() != 3)
    {
        return fail("unsupported context or latent model layout");
    }
    const auto latentPayload = reader.getPayload(model.children[3]);
    if (latentPayload.size() != 12 || readWord(latentPayload, 0) != 4 || readWord(latentPayload, 4) != 64 || std::bit_cast<float>(readWord(latentPayload, 8)) != 64.0f)
    {
        return fail("unsupported latent feature configuration");
    }
    const auto& embedding = nodes[context.children[3]];
    const auto& styleVectors = nodes[style.children[0]];
    if (embedding.type != "modl4" || embedding.children.size() != 1 || styleVectors.type != "cmpg1" || styleVectors.children.empty())
    {
        return fail("missing language embedding or default style vector");
    }

    AcousticModel candidate;
    if (auto result = candidate.features.load(reader, model.children[0]); result.failed())
    {
        return result;
    }
    if (candidate.features.getConfig().pitchChannels != 1 || candidate.features.getConfig().acousticChannels != 71 || candidate.features.getConfig().frameIntervalSeconds != 0.005f)
    {
        return fail("unsupported acoustic feature dimensions or frame interval");
    }
    if (auto result = reader.readFloatVector(model.children[1], candidate.speaker); result.failed())
    {
        return result;
    }
    if (auto result = reader.readFloatVector(styleVectors.children[0], candidate.defaultStyle); result.failed())
    {
        return result;
    }
    if (auto result = reader.readFloatMatrix(embedding.children[0], candidate.languageEmbedding); result.failed())
    {
        return result;
    }
    if (candidate.speaker.size() != 32 || candidate.defaultStyle.size() != 32 || candidate.languageEmbedding.rows != 8 || candidate.languageEmbedding.columns != candidate.features.getPhoneSets().size())
    {
        return fail("unsupported speaker, style or language embedding dimensions");
    }
    const std::array<std::pair<DnniInference*, std::size_t>, 10> networks{{
        {&candidate.phonemeEncoder, context.children[0]},
        {&candidate.contextProjection, context.children[1]},
        {&candidate.contextNetwork, context.children[2]},
        {&candidate.latentProjection, latent.children[0]},
        {&candidate.latentNetwork, latent.children[1]},
        {&candidate.latentHead, latent.children[2]},
        {&candidate.spectralNetwork, model.children[4]},
        {&candidate.spectralHead, model.children[5]},
        {&candidate.auxiliaryNetwork, model.children[6]},
        {&candidate.auxiliaryHead, model.children[7]},
    }};
    for (const auto& [network, node] : networks)
    {
        if (auto result = network->load(reader, node); result.failed())
        {
            return result;
        }
    }
    candidate.loaded = true;
    *this = std::move(candidate);
    return juce::Result::ok();
}
catch (const std::bad_alloc&)
{
    return fail("insufficient memory to load model parameters");
}

juce::Result AcousticModel::makeContext(std::span<const TimedPhoneme> phonemes, const DnniTensor& pitch, DnniTensor& output, State* state, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics) const
{
    DnniTensor phoneFeatures;
    if (auto result = features.encodePhonemes(phonemes, phoneFeatures); result.failed())
    {
        return result;
    }
    DnniTensor encoded;
    if (auto result = phonemeEncoder.run(phoneFeatures, encoded, nullptr, state != nullptr ? &state->phonemeEncoder : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(encoded, phonemes.size(), 64, "phoneme encoder"); result.failed())
    {
        return result;
    }
    DnniTensor context{pitch.frames, 144, std::vector<float>(pitch.frames * 144)};
    constexpr std::array<float, 4> phases{1.5707963705062866f, 0.0f, -1.5707963705062866f, -3.1415927410125732f};
    std::size_t frame = 0;
    for (std::size_t phone = 0; phone < phonemes.size(); ++phone)
    {
        const auto& phoneme = phonemes[phone];
        std::size_t language = 0;
        if (auto result = features.findLanguageIndex(phoneme.language, language); result.failed())
        {
            return result;
        }
        if (language >= languageEmbedding.columns)
        {
            return fail("language embedding index exceeds the stored matrix");
        }
        for (std::size_t withinPhone = 0; withinPhone < phoneme.frameCount; ++withinPhone, ++frame)
        {
            auto destination = context.values.begin() + static_cast<std::ptrdiff_t>(frame * 144);
            std::copy_n(encoded.values.begin() + static_cast<std::ptrdiff_t>(phone * 64), 64, destination);
            const float position = phoneme.frameCount == 1 ? 0.5f : static_cast<float>(withinPhone) / static_cast<float>(phoneme.frameCount - 1);
            const std::array<float, 2> scalars{position, pitch.values[frame]};
            for (std::size_t scalar = 0; scalar < scalars.size(); ++scalar)
            {
                const float angle = scalars[scalar] * 6.2831854820251465f;
                for (std::size_t phase = 0; phase < phases.size(); ++phase)
                {
                    destination[static_cast<std::ptrdiff_t>(64 + scalar * 4 + phase)] = static_cast<float>(static_cast<double>(std::cos(angle + phases[phase])) * 0.5 + 0.5);
                }
            }
            std::copy(speaker.begin(), speaker.end(), destination + 72);
            for (std::size_t channel = 0; channel < 8; ++channel)
            {
                destination[static_cast<std::ptrdiff_t>(104 + channel)] = languageEmbedding.values[channel * languageEmbedding.columns + language];
            }
            std::copy(defaultStyle.begin(), defaultStyle.end(), destination + 112);
        }
    }
    output = std::move(context);
    return juce::Result::ok();
}

juce::Result AcousticModel::sampleLatent(const DnniTensor& context, const DnniTensor& projected, std::uint32_t noiseSeed, DnniTensor& output, State* state, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics) const
{
    DnniTensor latentInput;
    if (auto result = latentProjection.run(projected, latentInput, nullptr, state != nullptr ? &state->latentProjection : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(latentInput, context.frames, 4, "latent projection"); result.failed())
    {
        return result;
    }
    DnniTensor condition{context.frames, 388, std::vector<float>(context.frames * 388)};
    std::array<float, 128> controls{};
    // The editor supplies these normal-take controls at 0x100090040 before DIDS;
    // the low-level empty-queue defaults (1, 1, 0) are not the editor defaults.
    constexpr float diffusionTemperature = 0.995f;
    constexpr float sourceTemperature = 0.5f;
    constexpr float guidance = 1.0f;
    for (std::size_t index = 0; index < 32; ++index)
    {
        const float frequency = std::exp2(static_cast<float>(index) * (-26.575424194335938f / 64.0f));
        const float diffusionAngle = 64.0f * diffusionTemperature * frequency;
        const float guidanceAngle = 64.0f * guidance * frequency;
        controls[index] = std::sin(diffusionAngle);
        controls[index + 32] = std::cos(diffusionAngle);
        controls[index + 64] = std::sin(guidanceAngle);
        controls[index + 96] = std::cos(guidanceAngle);
    }
    // This seed selects our reproducible noise sequence, not an original retake.
    GaussianNoise noise(noiseSeed);
    for (std::size_t frame = 0; frame < context.frames; ++frame)
    {
        auto destination = condition.values.begin() + static_cast<std::ptrdiff_t>(frame * 388);
        std::copy_n(context.values.begin() + static_cast<std::ptrdiff_t>(frame * 256), 256, destination);
        std::copy(controls.begin(), controls.end(), destination + 256);
        for (std::size_t channel = 0; channel < 4; ++channel)
        {
            destination[static_cast<std::ptrdiff_t>(384 + channel)] = noise.next() * sourceTemperature;
        }
    }
    DnniTensor hidden;
    if (auto result = latentNetwork.run(latentInput, hidden, &condition, state != nullptr ? &state->latentNetwork : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(hidden, context.frames, 128, "latent network"); result.failed())
    {
        return result;
    }
    return latentHead.run(hidden, output, nullptr, state != nullptr ? &state->latentHead : nullptr, shouldCancel, statistics);
}

juce::Result AcousticModel::run(std::span<const TimedPhoneme> phonemes, std::span<const float> logF0, DnniTensor& output, std::uint32_t noiseSeed, State* state, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics) const
try
{
    if (!loaded)
    {
        return fail("no model has been loaded");
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    std::size_t frames = 0;
    for (const auto& phoneme : phonemes)
    {
        if (phoneme.frameCount == 0 || phoneme.frameCount > maxFrames - frames)
        {
            return fail("phoneme frame counts must be positive and fit the inference memory limit");
        }
        frames += phoneme.frameCount;
    }
    if (frames == 0 || logF0.size() != frames)
    {
        return fail("one log-F0 value is required for each phoneme frame");
    }
    DnniTensor pitch;
    if (auto result = features.normalizeLogF0(logF0, pitch); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(pitch, frames, 1, "pitch normalization"); result.failed())
    {
        return result;
    }
    DnniTensor frameFeatures;
    if (auto result = makeContext(phonemes, pitch, frameFeatures, state, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    DnniTensor context;
    DnniTensor projected;
    if (auto result = contextNetwork.run(frameFeatures, context, nullptr, state != nullptr ? &state->contextNetwork : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = contextProjection.run(frameFeatures, projected, nullptr, state != nullptr ? &state->contextProjection : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(context, frames, 256, "context network"); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(projected, frames, 256, "context projection"); result.failed())
    {
        return result;
    }
    DnniTensor latent;
    if (auto result = sampleLatent(context, projected, noiseSeed, latent, state, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(latent, frames, 4, "latent head"); result.failed())
    {
        return result;
    }
    DnniTensor spectralHidden;
    DnniTensor spectrum;
    if (auto result = spectralNetwork.run(context, spectralHidden, &latent, state != nullptr ? &state->spectralNetwork : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = spectralHead.run(spectralHidden, spectrum, nullptr, state != nullptr ? &state->spectralHead : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(spectrum, frames, 64, "64-channel head"); result.failed())
    {
        return result;
    }
    DnniTensor auxiliaryHidden;
    DnniTensor auxiliary;
    if (auto result = auxiliaryNetwork.run(context, auxiliaryHidden, &spectrum, state != nullptr ? &state->auxiliaryNetwork : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = auxiliaryHead.run(auxiliaryHidden, auxiliary, nullptr, state != nullptr ? &state->auxiliaryHead : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(auxiliary, frames, 7, "7-channel head"); result.failed())
    {
        return result;
    }
    DnniTensor normalized{frames, 71, std::vector<float>(frames * 71)};
    for (std::size_t frame = 0; frame < frames; ++frame)
    {
        auto destination = normalized.values.begin() + static_cast<std::ptrdiff_t>(frame * 71);
        const auto source = auxiliary.values.begin() + static_cast<std::ptrdiff_t>(frame * 7);
        std::copy_n(source, 2, destination);
        std::copy_n(spectrum.values.begin() + static_cast<std::ptrdiff_t>(frame * 64), 64, destination + 2);
        std::copy_n(source + 2, 5, destination + 66);
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    return features.decodeAcoustic71(normalized, logF0, output);
}
catch (const std::bad_alloc&)
{
    return fail("insufficient memory for acoustic inference");
}
} // namespace sv::synthesis
