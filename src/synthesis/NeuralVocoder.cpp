#include "NeuralVocoder.h"

#include "PeriodicExcitation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace sv::synthesis
{
namespace
{
// The twelve aliases registered for the factory at 0x100171F60 in the reference.
constexpr std::array<std::uint64_t, 12> vocoderTypeIds = {
    0x6DFB67909B0414A1ULL,
    0xBC0ACD420E51B743ULL,
    0x1FD1749826B0CAECULL,
    0x88F2759EBD085477ULL,
    0x9B000CB774E2A953ULL,
    0x44D4CFABEA5DD27CULL,
    0x7963F08ABEB98D85ULL,
    0x950E9ACA274EE27DULL,
    0xA919046C636DB7B3ULL,
    0x7E6DCC7C7F9F8224ULL,
    0x9BC6670466293322ULL,
    0xB244CC8626A05CAEULL};

std::uint32_t readWord(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset]) | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

float readFloat(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return std::bit_cast<float>(readWord(bytes, offset));
}

juce::Result failure(const juce::String& message)
{
    return juce::Result::fail("Neural vocoder: " + message);
}

bool hasShape(const DnniTensor& tensor, std::size_t frames, std::size_t channels)
{
    return tensor.frames == frames && tensor.channels == channels && frames <= std::numeric_limits<std::size_t>::max() / channels && tensor.values.size() == frames * channels;
}
} // namespace

std::size_t NeuralVocoder::State::getBytes() const noexcept
{
    return shapeNetwork.getBytes() + conditionNetwork.getBytes() + modulationNetwork.getBytes() + spectralHeads.getBytes() + residual.getBytes();
}

juce::Result NeuralVocoder::load(const DnniReader& reader, std::size_t rootNode)
{
    const auto& nodes = reader.getNodes();
    if (rootNode >= nodes.size())
    {
        return failure("root node is out of range");
    }

    const auto& node = nodes[rootNode];
    if (std::find(vocoderTypeIds.begin(), vocoderTypeIds.end(), node.typeId) == vocoderTypeIds.end())
    {
        return failure("unsupported root type at offset " + juce::String(node.offset));
    }
    const auto payload = reader.getPayload(rootNode);
    if (payload.size() != 18 || node.children.size() != 13 || (payload[12] & 1U) == 0)
    {
        return failure("expected the 13-child architecture with a shape network");
    }

    const auto configurationNode = node.children[0];
    const auto& configuration = nodes[configurationNode];
    const auto configurationBytes = reader.getPayload(configurationNode);
    const bool featureVersion2 = configuration.type == "_vocfv2";
    if ((configuration.type != "_vocfv1" && !featureVersion2) || configurationBytes.size() != (featureVersion2 ? 52 : 48) || configuration.children.size() < 4)
    {
        return failure("invalid _vocfv1/v2 feature configuration");
    }
    if (readWord(configurationBytes, 0) != 72 || readWord(configurationBytes, 16) != 72 || readWord(configurationBytes, 32) != 1)
    {
        return failure("unsupported feature dimensions or F0 expansion");
    }

    NeuralVocoder candidate;
    // _vocfv2 adds the input voicing override used by 0x1002127e0 before
    // normalization; the first twelve fields retain the _vocfv1 layout.
    candidate.forceVoicedFeatures = featureVersion2 && readWord(configurationBytes, 48) != 0;
    candidate.sampleRate = static_cast<double>(readWord(configurationBytes, 8));
    candidate.framePeriodSeconds = static_cast<double>(readFloat(configurationBytes, 12));
    const double hop = std::round(candidate.sampleRate * candidate.framePeriodSeconds);
    candidate.excitationGain = std::exp(readFloat(payload, 4));
    candidate.modulationBias = readFloat(payload, 14);
    if (!std::isfinite(candidate.sampleRate) || candidate.sampleRate <= 0.0 || !std::isfinite(candidate.framePeriodSeconds) || candidate.framePeriodSeconds <= 0.0 || !std::isfinite(hop) || hop < 1.0 || hop > static_cast<double>(std::numeric_limits<int>::max()) || !std::isfinite(candidate.excitationGain) || candidate.excitationGain <= 0.0f || !std::isfinite(candidate.modulationBias))
    {
        return failure("invalid sample timing or gain configuration");
    }
    candidate.hopSamples = static_cast<std::size_t>(hop);

    auto result = candidate.inputNormalizer.load(reader, configuration.children[0]);
    if (result.failed())
    {
        return result;
    }
    if (candidate.inputNormalizer.getChannelCount() != 72)
    {
        return failure("input normalization must have 72 channels");
    }

    result = candidate.shapeNetwork.load(reader, node.children[1]);
    if (result.failed())
    {
        return result;
    }
    result = candidate.conditionNetwork.load(reader, node.children[2]);
    if (result.failed())
    {
        return result;
    }
    result = candidate.spectralHeads.load(reader, node.children[3]);
    if (result.failed())
    {
        return result;
    }
    result = reader.readFloatVector(node.children[4], candidate.modulationRatios);
    if (result.failed())
    {
        return result;
    }
    if (candidate.modulationRatios.empty() || !std::all_of(candidate.modulationRatios.begin(), candidate.modulationRatios.end(), [](float value)
                                                           { return std::isfinite(value) && value > 0.0f; }))
    {
        return failure("invalid modulation frequency ratios");
    }
    result = candidate.modulationNetwork.load(reader, node.children[5]);
    if (result.failed())
    {
        return result;
    }
    for (std::size_t index = 0; index < candidate.filters.size(); ++index)
    {
        result = candidate.filters[index].load(reader, node.children[6 + index], candidate.hopSamples);
        if (result.failed())
        {
            return result;
        }
    }
    result = candidate.residual.load(reader, rootNode);
    if (result.failed())
    {
        return result;
    }
    candidate.loaded = true;
    *this = std::move(candidate);
    return juce::Result::ok();
}

juce::Result NeuralVocoder::run(const DnniTensor& input, NeuralVocoderOutput& output, std::span<const float> shapeControl, SynthesisStatistics* statistics, State* state, const std::function<bool()>& shouldCancel, DnniRunStatistics* networkStatistics) const
{
    const double started = juce::Time::getMillisecondCounterHiRes();
    const juce::ScopeGuard finishStatistics([statistics, started]
                                            {
        if (statistics != nullptr)
        {
            statistics->vocoderMilliseconds += juce::Time::getMillisecondCounterHiRes() - started;
        } });
    if (!loaded)
    {
        return failure("no model loaded");
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    if (input.frames == 0 || !hasShape(input, input.frames, 72) || input.frames > std::numeric_limits<std::size_t>::max() / hopSamples)
    {
        return failure("expected a nonempty frame-major tensor with 72 channels");
    }
    if ((!shapeControl.empty() && shapeControl.size() != input.frames) || !std::all_of(shapeControl.begin(), shapeControl.end(), [](float value)
                                                                                       { return std::isfinite(value); }))
    {
        return failure("shape control must be empty or contain one finite value per frame");
    }
    if (!std::all_of(input.values.begin(), input.values.end(), [](float value)
                     { return std::isfinite(value); }))
    {
        return failure("input contains nonfinite values");
    }

    double stageStarted = juce::Time::getMillisecondCounterHiRes();
    DnniTensor logarithmic = input;
    for (std::size_t frame = 0; frame < input.frames; ++frame)
    {
        const float frequency = input.values[frame * 72 + 1];
        if (frequency <= 0.0f)
        {
            return failure("F0 must be positive even for unvoiced frames");
        }
        if (forceVoicedFeatures)
        {
            logarithmic.values[frame * 72] = 1.0f;
        }
        logarithmic.values[frame * 72 + 1] = std::log(frequency);
    }

    DnniTensor normalized;
    auto result = inputNormalizer.normalize(logarithmic, normalized, true);
    if (result.failed())
    {
        return result;
    }
    DnniTensor shape;
    result = shapeNetwork.run(normalized, shape, nullptr, state != nullptr ? &state->shapeNetwork : nullptr, shouldCancel, networkStatistics);
    if (result.failed())
    {
        return result;
    }
    if (!hasShape(shape, input.frames, 1))
    {
        return failure("shape network output does not align with input frames");
    }
    std::vector<PeriodicExcitationFrame> sourceFrames(input.frames);
    for (std::size_t frame = 0; frame < input.frames; ++frame)
    {
        // The shape head emits a logit; a larger value must increase Rd.
        float sourceShape = 2.9f / (std::exp(-shape.values[frame]) + 1.0f) + 0.1f;
        if (!shapeControl.empty())
        {
            const float control = shapeControl[frame];
            sourceShape += control * (control <= 0.0f ? sourceShape - 0.1f : 3.0f - sourceShape);
        }
        sourceFrames[frame] = {input.values[frame * 72 + 1], sourceShape, forceVoicedFeatures || input.values[frame * 72] >= 0.5f};
    }

    DnniTensor condition;
    result = conditionNetwork.run(normalized, condition, nullptr, state != nullptr ? &state->conditionNetwork : nullptr, shouldCancel, networkStatistics);
    if (result.failed())
    {
        return result;
    }
    std::array<DnniTensor, 3> spectra;
    result = spectralHeads.run(condition, spectra, state != nullptr ? &state->spectralHeads : nullptr, shouldCancel, networkStatistics);
    if (result.failed())
    {
        return result;
    }
    DnniTensor modulationParameters;
    result = modulationNetwork.run(condition, modulationParameters, nullptr, state != nullptr ? &state->modulationNetwork : nullptr, shouldCancel, networkStatistics);
    if (result.failed())
    {
        return result;
    }
    if (modulationRatios.size() > std::numeric_limits<std::size_t>::max() / 2 || !hasShape(modulationParameters, input.frames, modulationRatios.size() * 2))
    {
        return failure("modulation network output does not align with input frames");
    }

    if (statistics != nullptr)
    {
        statistics->vocoderNetworkMilliseconds += juce::Time::getMillisecondCounterHiRes() - stageStarted;
    }
    stageStarted = juce::Time::getMillisecondCounterHiRes();
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    std::vector<float> excitation;
    std::vector<double> phaseCycles;
    result = renderPeriodicExcitation(sourceFrames, hopSamples, framePeriodSeconds, excitationGain, excitation, phaseCycles);
    if (result.failed())
    {
        return result;
    }
    std::vector<float> modulation;
    result = renderPeriodicModulation(sourceFrames, phaseCycles, modulationRatios, modulationParameters.values, hopSamples, framePeriodSeconds, modulation);
    if (result.failed())
    {
        return result;
    }
    const std::size_t sampleCount = input.frames * hopSamples;
    if (excitation.size() != sampleCount || modulation.size() != sampleCount)
    {
        return failure("periodic source length does not match frame timing");
    }
    for (std::size_t sample = 0; sample < sampleCount; ++sample)
    {
        modulation[sample] = excitation[sample] * (modulation[sample] + modulationBias);
    }

    if (statistics != nullptr)
    {
        statistics->vocoderSourceMilliseconds += juce::Time::getMillisecondCounterHiRes() - stageStarted;
    }
    stageStarted = juce::Time::getMillisecondCounterHiRes();
    std::vector<float> residualExcitation;
    result = residual.render(normalized, excitation, residualExcitation, state != nullptr ? &state->residual : nullptr, shouldCancel, networkStatistics);
    if (result.failed())
    {
        return result;
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    if (statistics != nullptr)
    {
        statistics->vocoderResidualMilliseconds += juce::Time::getMillisecondCounterHiRes() - stageStarted;
    }
    stageStarted = juce::Time::getMillisecondCounterHiRes();
    std::array<std::vector<float>, 3> components;
    result = filters[0].render(spectra[0], excitation, components[0]);
    if (result.failed())
    {
        return result;
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    result = filters[1].render(spectra[1], residualExcitation, components[1]);
    if (result.failed())
    {
        return result;
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    result = filters[2].render(spectra[2], modulation, components[2]);
    if (result.failed())
    {
        return result;
    }
    for (const auto& component : components)
    {
        if (component.size() != sampleCount)
        {
            return failure("filtered output length does not match frame timing");
        }
    }

    NeuralVocoderOutput candidate;
    candidate.sampleRate = sampleRate;
    candidate.hopSamples = hopSamples;
    candidate.samples.resize(sampleCount);
    candidate.periodic.resize(sampleCount);
    candidate.aperiodic = std::move(components[1]);
    for (std::size_t sample = 0; sample < sampleCount; ++sample)
    {
        candidate.periodic[sample] = components[0][sample] + components[2][sample];
        candidate.samples[sample] = components[0][sample] + candidate.aperiodic[sample] + components[2][sample];
        if (!std::isfinite(candidate.samples[sample]) || !std::isfinite(candidate.periodic[sample]))
        {
            return failure("nonfinite PCM generated by the model");
        }
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    output = std::move(candidate);
    if (statistics != nullptr)
    {
        statistics->vocoderFilterMilliseconds += juce::Time::getMillisecondCounterHiRes() - stageStarted;
        statistics->vocoderFrames += input.frames;
    }
    return juce::Result::ok();
}
} // namespace sv::synthesis
