#include "PitchContext.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

namespace sv::synthesis
{
namespace
{
constexpr std::size_t contextChannels = 328;
constexpr std::size_t maximumFrames = 64 * 1024 * 1024 / contextChannels;

juce::Result fail(const juce::String& message)
{
    return juce::Result::fail("Pitch context: " + message);
}

juce::Result checkShape(const DnniTensor& tensor, std::size_t frames, std::size_t channels)
{
    if (tensor.frames != frames || tensor.channels != channels || tensor.values.size() != frames * channels)
    {
        return fail("unexpected tensor dimensions");
    }
    if (!std::all_of(tensor.values.begin(), tensor.values.end(), [](float value)
                     { return std::isfinite(value); }))
    {
        return fail("non-finite tensor value");
    }
    return juce::Result::ok();
}

juce::Result countFrames(std::span<const std::size_t> durations, std::size_t& total)
{
    total = 0;
    for (const auto duration : durations)
    {
        if (duration > maximumFrames - total)
        {
            return fail("frame sequence exceeds the memory limit");
        }
        total += duration;
    }
    return juce::Result::ok();
}
} // namespace

std::size_t PitchContext::State::getBytes() const noexcept
{
    return projection.getBytes() + feedForward.getBytes() + residual.getBytes();
}

juce::Result PitchContext::load(const DnniReader& reader, std::size_t nodeIndex, std::size_t phonemeCategoryCount, std::size_t languageCount)
try
{
    const auto& nodes = reader.getNodes();
    constexpr std::array<std::uint64_t, 12> contextTypes{0xa5e321251caa33ed, 0xef76985afa86ddf3, 0xad6ece198010cb64, 0x2b04774298206d1f, 0xf396d3b2a8582243, 0x01771a1f6fab6574, 0xb1ec6d6128b14ae9, 0xfa28a01cd2e6d271, 0xb092f5e4e64d93a3, 0x470db6b41577b68c, 0x8ef5b03449c54d0e, 0x3ff67ea05f320122};
    if (nodeIndex >= nodes.size() || std::find(contextTypes.begin(), contextTypes.end(), nodes[nodeIndex].typeId) == contextTypes.end() || nodes[nodeIndex].children.size() != 7 || nodes[nodeIndex].payloadSize != 4)
    {
        return fail("unsupported gen5 context layout");
    }
    const auto& children = nodes[nodeIndex].children;
    const auto& phoneEmbeddingNode = nodes[children[0]];
    const auto& languageEmbeddingNode = nodes[children[1]];
    if (phoneEmbeddingNode.type != "modl4" || phoneEmbeddingNode.children.size() != 1 || languageEmbeddingNode.type != "modl4" || languageEmbeddingNode.children.size() != 1)
    {
        return fail("missing categorical embeddings");
    }
    PitchContext candidate;
    if (auto result = reader.readFloatMatrix(phoneEmbeddingNode.children[0], candidate.phonemeEmbedding); result.failed())
    {
        return result;
    }
    if (auto result = reader.readFloatMatrix(languageEmbeddingNode.children[0], candidate.languageEmbedding); result.failed())
    {
        return result;
    }
    if (phonemeCategoryCount == 0 || languageCount == 0 || candidate.phonemeEmbedding.rows != 32 || candidate.phonemeEmbedding.columns != phonemeCategoryCount || candidate.languageEmbedding.rows != 32 || candidate.languageEmbedding.columns != languageCount)
    {
        return fail("32-channel categorical embeddings must match the frontend's category and language tables");
    }
    if (auto result = candidate.noteEncoder.load(reader, children[2]); result.failed())
    {
        return result;
    }
    if (auto result = candidate.phonemeEncoder.load(reader, children[3]); result.failed())
    {
        return result;
    }
    if (auto result = candidate.projection.load(reader, children[4]); result.failed())
    {
        return result;
    }
    if (auto result = candidate.feedForward.load(reader, children[5]); result.failed())
    {
        return result;
    }
    if (auto result = candidate.residual.load(reader, children[6]); result.failed())
    {
        return result;
    }
    candidate.loaded = true;
    *this = std::move(candidate);
    return juce::Result::ok();
}
catch (const std::bad_alloc&)
{
    return fail("not enough memory to load the model");
}

juce::Result PitchContext::run(const PitchFeatureOutput& features, std::span<const float> speaker, DnniTensor& feedForwardOutput, DnniTensor& residualOutput, State* state, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics) const
try
{
    if (!loaded)
    {
        return fail("model has not been loaded");
    }
    const auto noteCount = features.noteFrameCounts.size();
    const auto phoneCount = features.phonemeFrameCounts.size();
    if (noteCount == 0 || phoneCount == 0 || noteCount > maximumFrames || phoneCount > maximumFrames || features.phonemeCategories.size() != phoneCount || features.phonemeLanguages.size() != phoneCount)
    {
        return fail("inconsistent score or phoneme sequences");
    }
    if (auto result = checkShape(features.noteFeatures, noteCount, 10); result.failed())
    {
        return result;
    }
    if (speaker.size() != 32 || !std::all_of(speaker.begin(), speaker.end(), [](float value)
                                             { return std::isfinite(value); }))
    {
        return fail("expected a finite 32-channel speaker vector");
    }
    std::size_t noteFrames = 0;
    std::size_t phoneFrames = 0;
    if (auto result = countFrames(features.noteFrameCounts, noteFrames); result.failed())
    {
        return result;
    }
    if (auto result = countFrames(features.phonemeFrameCounts, phoneFrames); result.failed())
    {
        return result;
    }
    const auto frameCount = std::max(noteFrames, phoneFrames);
    if (frameCount == 0)
    {
        return fail("empty prediction interval");
    }
    if (shouldCancel && shouldCancel())
    {
        return fail("cancelled");
    }
    DnniTensor noteContext;
    if (auto result = noteEncoder.run(features.noteFeatures, noteContext, nullptr, nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(noteContext, noteCount, 128); result.failed())
    {
        return result;
    }
    DnniTensor phoneInput{phoneCount, 32, std::vector<float>(phoneCount * 32)};
    for (std::size_t phone = 0; phone < phoneCount; ++phone)
    {
        const auto category = features.phonemeCategories[phone];
        if (category >= phonemeEmbedding.columns || features.phonemeLanguages[phone] >= languageEmbedding.columns)
        {
            return fail("categorical index exceeds the embedding dimensions");
        }
        for (std::size_t channel = 0; channel < 32; ++channel)
        {
            phoneInput.values[phone * 32 + channel] = phonemeEmbedding.values[channel * phonemeEmbedding.columns + category];
        }
    }
    DnniTensor phoneContext;
    if (auto result = phonemeEncoder.run(phoneInput, phoneContext, nullptr, nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(phoneContext, phoneCount, 128); result.failed())
    {
        return result;
    }
    // The original context builder uses the longer expanded sequence and leaves
    // the missing recurrent channels zero when the two rounding grids differ.
    DnniTensor context{frameCount, contextChannels, std::vector<float>(frameCount * contextChannels, 0.0f)};
    std::size_t frame = 0;
    for (std::size_t note = 0; note < noteCount; ++note)
    {
        for (std::size_t withinNote = 0; withinNote < features.noteFrameCounts[note]; ++withinNote, ++frame)
        {
            if ((frame & 255) == 0 && shouldCancel && shouldCancel())
            {
                return fail("cancelled");
            }
            std::copy_n(noteContext.values.data() + note * 128, 128, context.values.data() + frame * contextChannels);
        }
    }
    frame = 0;
    std::size_t lastLanguage = 0;
    for (std::size_t phone = 0; phone < phoneCount; ++phone)
    {
        const auto duration = features.phonemeFrameCounts[phone];
        for (std::size_t withinPhone = 0; withinPhone < duration; ++withinPhone, ++frame)
        {
            if ((frame & 255) == 0 && shouldCancel && shouldCancel())
            {
                return fail("cancelled");
            }
            auto* destination = context.values.data() + frame * contextChannels;
            std::copy_n(phoneContext.values.data() + phone * 128, 128, destination + 128);
            const float position = duration == 1 ? 0.5f : static_cast<float>(withinPhone) / static_cast<float>(duration - 1);
            for (std::size_t phase = 0; phase < 8; ++phase)
            {
                const float angle = (position + (1.0f - static_cast<float>(phase)) * 0.125f) * 6.2831854820251465f;
                destination[256 + phase] = static_cast<float>(static_cast<double>(std::cos(angle)) * 0.5 + 0.5);
            }
            lastLanguage = features.phonemeLanguages[phone];
            for (std::size_t channel = 0; channel < 32; ++channel)
            {
                destination[296 + channel] = languageEmbedding.values[channel * languageEmbedding.columns + lastLanguage];
            }
        }
    }
    for (; frame < frameCount; ++frame)
    {
        for (std::size_t channel = 0; channel < 32; ++channel)
        {
            context.values[frame * contextChannels + 296 + channel] = languageEmbedding.values[channel * languageEmbedding.columns + lastLanguage];
        }
    }
    for (frame = 0; frame < frameCount; ++frame)
    {
        std::copy(speaker.begin(), speaker.end(), context.values.begin() + static_cast<std::ptrdiff_t>(frame * contextChannels + 264));
    }
    DnniTensor projected;
    if (auto result = projection.run(context, projected, nullptr, state != nullptr ? &state->projection : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(projected, frameCount, 128); result.failed())
    {
        return result;
    }
    DnniTensor feedForwardResult;
    DnniTensor residualResult;
    if (auto result = feedForward.run(projected, feedForwardResult, nullptr, state != nullptr ? &state->feedForward : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = residual.run(projected, residualResult, nullptr, state != nullptr ? &state->residual : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(feedForwardResult, frameCount, 128); result.failed())
    {
        return result;
    }
    if (auto result = checkShape(residualResult, frameCount, 128); result.failed())
    {
        return result;
    }
    feedForwardOutput = std::move(feedForwardResult);
    residualOutput = std::move(residualResult);
    return juce::Result::ok();
}
catch (const std::bad_alloc&)
{
    return fail("not enough memory to evaluate the model");
}
} // namespace sv::synthesis
