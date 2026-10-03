#include "VocoderResidual.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <string_view>
#include <utility>

namespace sv::synthesis
{
namespace
{
constexpr std::size_t maximumValues = 64 * 1024 * 1024;
constexpr std::array<std::uint64_t, 12> vocoderTypeIds{
    0x6dfb67909b0414a1ULL, 0xbc0acd420e51b743ULL, 0x1fd1749826b0caecULL, 0x88f2759ebd085477ULL, 0x9b000cb774e2a953ULL, 0x44d4cfabea5dd27cULL, 0x7963f08abeb98d85ULL, 0x950e9aca274ee27dULL, 0xa919046c636db7b3ULL, 0x7e6dcc7c7f9f8224ULL, 0x9bc6670466293322ULL, 0xb244cc8626a05caeULL};

std::uint32_t readWord(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset]) | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

juce::Result loadFilters(const DnniReader& reader, std::size_t nodeIndex, std::string_view expectedType, std::size_t bands, std::vector<std::vector<float>>& filters)
{
    const auto& nodes = reader.getNodes();
    if (nodeIndex >= nodes.size())
    {
        return juce::Result::fail("Vocoder residual: filter-bank node is out of range.");
    }
    const auto& node = nodes[nodeIndex];
    if (node.type != expectedType || node.payloadSize != 0 || node.children.size() != bands)
    {
        return juce::Result::fail("Vocoder residual: resampler must contain one coefficient vector per band.");
    }

    filters.resize(bands);
    for (std::size_t band = 0; band < bands; ++band)
    {
        const auto result = reader.readFloatVector(node.children[band], filters[band]);
        if (result.failed())
        {
            return result;
        }
        const auto& values = filters[band];
        if (values.size() < 3 || values.size() > 4097 || values.size() % 2 == 0 || values.size() != filters.front().size() || !std::all_of(values.begin(), values.end(), [](float value)
                                                                                                                                           { return std::isfinite(value); }))
        {
            return juce::Result::fail("Vocoder residual: expected finite, equal-length odd FIR coefficient vectors.");
        }
    }
    return juce::Result::ok();
}

class GaussianExcitation
{
public:
    float next()
    {
        if (hasSpare)
        {
            hasSpare = false;
            return spare;
        }

        float first = 0.0f;
        float second = 0.0f;
        float radiusSquared = 0.0f;
        do
        {
            first = static_cast<float>(generator() - 1) * 0x1p-30f - 1.0f;
            second = static_cast<float>(generator() - 1) * 0x1p-30f - 1.0f;
            radiusSquared = second * second + first * first;
        } while (radiusSquared > 1.0f || radiusSquared == 0.0f);

        const float scale = std::sqrt(-2.0f * std::log(radiusSquared) / radiusSquared);
        spare = second * scale;
        hasSpare = true;
        return first * scale;
    }

private:
    std::minstd_rand generator{1};
    float spare = 0.0f;
    bool hasSpare = false;
};

juce::Result analyseExcitation(std::span<const float> input, const std::vector<std::vector<float>>& filters, std::size_t expectedFrames, DnniTensor& output)
{
    using Vector = juce::dsp::SIMDRegister<float>;
    const auto bands = filters.size();
    const auto filterLength = filters.front().size();
    const auto tailSamples = filterLength / 2;
    const auto discardedFrames = bands / 2 - 1;
    const auto firstSample = bands - 1 - tailSamples % bands;
    const auto sampleCount = input.size() + tailSamples;
    const auto emittedFrames = sampleCount > firstSample ? 1 + (sampleCount - firstSample - 1) / bands : 0;
    if (emittedFrames != expectedFrames + discardedFrames + 1)
    {
        return juce::Result::fail("Vocoder residual: analysis-filter timing does not match the model's subband frames.");
    }

    // Each lane accumulates an independent band in the original tap order.
    const auto bandBlocks = (bands + Vector::size() - 1) / Vector::size();
    std::vector<Vector> coefficients(bandBlocks * filterLength);
    alignas(Vector) std::array<float, Vector::size()> lanes{};
    for (std::size_t block = 0; block < bandBlocks; ++block)
    {
        const auto firstBand = block * Vector::size();
        const auto blockBands = std::min(Vector::size(), bands - firstBand);
        lanes.fill(0.0f);
        for (std::size_t coefficient = 0; coefficient < filterLength; ++coefficient)
        {
            for (std::size_t lane = 0; lane < blockBands; ++lane)
            {
                lanes[lane] = filters[firstBand + lane][coefficient];
            }
            coefficients[block * filterLength + coefficient] = Vector::fromRawArray(lanes.data());
        }
    }

    GaussianExcitation noise;
    output.frames = expectedFrames;
    output.channels = bands * 2;
    output.values.resize(expectedFrames * output.channels);

    // The analysis operator starts with half a filter of zero history. Its
    // caller discards the first bands/2-1 frames and the final frame.
    for (std::size_t frame = 0; frame < expectedFrames; ++frame)
    {
        const auto sample = firstSample + (frame + discardedFrames) * bands;
        const auto historyStart = static_cast<std::int64_t>(sample) - static_cast<std::int64_t>(filterLength - 1);
        const auto coefficientBegin = static_cast<std::size_t>(std::max<std::int64_t>(0, -historyStart));
        const auto coefficientEnd = static_cast<std::size_t>(std::min<std::int64_t>(static_cast<std::int64_t>(filterLength), static_cast<std::int64_t>(input.size()) - historyStart));
        const auto sourceBegin = static_cast<std::size_t>(historyStart + static_cast<std::int64_t>(coefficientBegin));
        auto* destination = output.values.data() + frame * output.channels;
        for (std::size_t block = 0; block < bandBlocks; ++block)
        {
            auto sum = Vector::expand(0.0f);
            const auto* source = input.data() + sourceBegin;
            const auto* weights = coefficients.data() + block * filterLength;
            for (std::size_t coefficient = coefficientBegin; coefficient < coefficientEnd; ++coefficient)
            {
                sum += weights[coefficient] * Vector::expand(*source++);
            }
            sum.copyToRawArray(lanes.data());
            const auto firstBand = block * Vector::size();
            std::copy_n(lanes.data(), std::min(Vector::size(), bands - firstBand), destination + firstBand);
        }
        for (std::size_t band = 0; band < bands; ++band)
        {
            destination[bands + band] = noise.next();
        }
    }
    return juce::Result::ok();
}

DnniTensor interpolateConditions(const DnniTensor& input, std::size_t factor)
{
    DnniTensor output;
    output.frames = input.frames * factor;
    output.channels = input.channels;
    output.values.resize(output.frames * output.channels);
    const auto leadingFrames = factor / 2;
    const auto trailingFrames = factor - leadingFrames;
    const float offset = factor % 2 == 0 ? 0.5f : 0.0f;
    const float inverseFactor = 1.0f / static_cast<float>(factor);
    std::size_t destination = 0;
    for (std::size_t repeat = 0; repeat < leadingFrames; ++repeat)
    {
        std::copy_n(input.values.begin(), input.channels, output.values.begin() + static_cast<std::ptrdiff_t>(destination));
        destination += input.channels;
    }
    for (std::size_t frame = 1; frame < input.frames; ++frame)
    {
        for (std::size_t step = 0; step < factor; ++step)
        {
            const float position = static_cast<float>(step) + offset;
            for (std::size_t channel = 0; channel < input.channels; ++channel)
            {
                const float previous = input.values[(frame - 1) * input.channels + channel];
                const float current = input.values[frame * input.channels + channel];
                output.values[destination++] = (current - previous) * position * inverseFactor + previous;
            }
        }
    }
    for (std::size_t repeat = 0; repeat < trailingFrames; ++repeat)
    {
        std::copy_n(input.values.end() - static_cast<std::ptrdiff_t>(input.channels), input.channels, output.values.begin() + static_cast<std::ptrdiff_t>(destination));
        destination += input.channels;
    }
    return output;
}

std::vector<float> synthesiseSubbands(const DnniTensor& input, const std::vector<std::vector<float>>& filters)
{
    const auto filterLength = filters.front().size();
    const auto latencySamples = (filterLength - 1) / 2;
    std::vector<float> ring(std::max<std::size_t>(512, filterLength), 0.0f);
    std::vector<float> output;
    output.reserve(input.frames * input.channels);
    std::size_t position = 0;
    std::size_t pendingLatency = 0;
    const auto emit = [&]()
    {
        output.push_back(ring[position]);
        ring[position] = 0.0f;
        if (++position == ring.size())
        {
            position = 0;
        }
    };

    for (std::size_t frame = 0; frame < input.frames; ++frame)
    {
        const auto firstSegment = std::min(filterLength, ring.size() - position);
        const auto secondSegment = filterLength - firstSegment;
        for (std::size_t band = 0; band < input.channels; ++band)
        {
            const float amplitude = input.values[frame * input.channels + band];
            juce::FloatVectorOperations::addWithMultiply(ring.data() + position, filters[band].data(), amplitude, static_cast<int>(firstSegment));
            if (secondSegment != 0)
            {
                juce::FloatVectorOperations::addWithMultiply(ring.data(), filters[band].data() + firstSegment, amplitude, static_cast<int>(secondSegment));
            }
        }
        for (std::size_t sample = 0; sample < input.channels; ++sample)
        {
            // The reference increments this initial delay without advancing
            // the accumulation cursor; retaining that boundary behavior is
            // necessary for the serialized synthesis filters.
            if (pendingLatency < latencySamples)
            {
                ++pendingLatency;
            }
            else
            {
                emit();
            }
        }
    }
    while (pendingLatency > 0)
    {
        emit();
        --pendingLatency;
    }
    return output;
}
} // namespace

std::size_t VocoderResidual::State::getBytes() const noexcept
{
    return network.getBytes() + projection.getBytes();
}

juce::Result VocoderResidual::load(const DnniReader& reader, std::size_t vocoderRoot)
{
    loaded = false;
    const auto& nodes = reader.getNodes();
    if (vocoderRoot >= nodes.size() || std::find(vocoderTypeIds.begin(), vocoderTypeIds.end(), nodes[vocoderRoot].typeId) == vocoderTypeIds.end())
    {
        return juce::Result::fail("Vocoder residual: root node is out of range or has an unsupported type.");
    }
    const auto rootPayload = reader.getPayload(vocoderRoot);
    if (rootPayload.size() != 18 || rootPayload[12] > 1 || rootPayload[13] > 1)
    {
        return juce::Result::fail("Vocoder residual: unsupported vocoder root configuration.");
    }
    const auto firstNetwork = rootPayload[12] != 0 ? std::size_t{2} : std::size_t{1};
    const auto& children = nodes[vocoderRoot].children;
    if (children.size() != firstNetwork + 11)
    {
        return juce::Result::fail("Vocoder residual: vocoder root has an unexpected child count.");
    }
    const auto configuration = reader.getPayload(children.front());
    const auto& configurationType = nodes[children.front()].type;
    const bool featureVersion2 = configurationType == "_vocfv2";
    if ((configurationType != "_vocfv1" && !featureVersion2) || configuration.size() != (featureVersion2 ? 52 : 48))
    {
        return juce::Result::fail("Vocoder residual: missing vocoder feature configuration.");
    }
    const auto bandCount = static_cast<std::size_t>(readWord(rootPayload, 0));
    const auto channels = readWord(configuration, 0);
    const auto sampleRate = readWord(configuration, 8);
    const float framePeriod = std::bit_cast<float>(readWord(configuration, 12));
    const float hop = std::round(static_cast<float>(sampleRate) * framePeriod);
    if (bandCount < 2 || bandCount > 64 || bandCount % 2 != 0 || channels == 0 || channels > 4096 || !std::isfinite(hop) || hop < 1.0f || hop > 65536.0f || static_cast<std::size_t>(hop) % bandCount != 0)
    {
        return juce::Result::fail("Vocoder residual: invalid sample rate, frame period, or band count.");
    }

    auto result = loadFilters(reader, children[firstNetwork + 7], "_ppusv0", bandCount, synthesisFilters);
    if (result.failed())
    {
        return result;
    }
    result = loadFilters(reader, children[firstNetwork + 8], "_ppdsv0", bandCount, analysisFilters);
    if (result.failed())
    {
        return result;
    }
    result = network.load(reader, children[firstNetwork + 9]);
    if (result.failed())
    {
        return result;
    }
    result = projection.load(reader, children[firstNetwork + 10]);
    if (result.failed())
    {
        return result;
    }
    hopSamples = static_cast<std::size_t>(hop);
    bands = bandCount;
    featureChannels = channels;
    gateNoise = rootPayload[13] != 0;
    loaded = true;
    return juce::Result::ok();
}

juce::Result VocoderResidual::render(const DnniTensor& normalizedFeatures, std::span<const float> excitation, std::vector<float>& output, State* state, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics) const
{
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    if (!loaded)
    {
        return juce::Result::fail("Vocoder residual: no model has been loaded.");
    }
    if (normalizedFeatures.channels != featureChannels || normalizedFeatures.frames > maximumValues / std::max<std::size_t>(featureChannels, 1) || normalizedFeatures.values.size() != normalizedFeatures.frames * featureChannels || normalizedFeatures.frames > maximumValues / hopSamples || excitation.size() != normalizedFeatures.frames * hopSamples)
    {
        return juce::Result::fail("Vocoder residual: condition frames and excitation length do not match the model.");
    }
    if (!std::all_of(excitation.begin(), excitation.end(), [](float value)
                     { return std::isfinite(value); }) ||
        !std::all_of(normalizedFeatures.values.begin(), normalizedFeatures.values.end(), [](float value)
                     { return std::isfinite(value); }))
    {
        return juce::Result::fail("Vocoder residual: input contains a non-finite value.");
    }
    if (normalizedFeatures.frames == 0)
    {
        output.clear();
        return juce::Result::ok();
    }
    const auto interpolationFactor = hopSamples / bands;
    const auto subbandFrames = normalizedFeatures.frames * interpolationFactor;
    if (featureChannels == 0 || subbandFrames > maximumValues / featureChannels || subbandFrames > maximumValues / (bands * 2))
    {
        return juce::Result::fail("Vocoder residual: expanded condition tensor exceeds the memory limit.");
    }

    DnniTensor analysis;
    auto result = analyseExcitation(excitation, analysisFilters, subbandFrames, analysis);
    if (result.failed())
    {
        return result;
    }
    const auto condition = interpolateConditions(normalizedFeatures, interpolationFactor);
    DnniTensor hidden;
    result = network.run(analysis, hidden, &condition, state != nullptr ? &state->network : nullptr, shouldCancel, statistics);
    if (result.failed())
    {
        return result;
    }
    DnniTensor subbands;
    result = projection.run(hidden, subbands, nullptr, state != nullptr ? &state->projection : nullptr, shouldCancel, statistics);
    if (result.failed())
    {
        return result;
    }
    if (subbands.frames != subbandFrames || subbands.channels != bands)
    {
        return juce::Result::fail("Vocoder residual: projected subband dimensions do not match the model.");
    }
    if (gateNoise)
    {
        for (std::size_t frame = 0; frame < subbandFrames; ++frame)
        {
            for (std::size_t band = 0; band < bands; ++band)
            {
                const auto index = frame * bands + band;
                const float noise = analysis.values[frame * bands * 2 + bands + band];
                subbands.values[index] = noise / (1.0f + std::exp(-subbands.values[index]));
            }
        }
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    auto samples = synthesiseSubbands(subbands, synthesisFilters);
    if (samples.size() != excitation.size() || !std::all_of(samples.begin(), samples.end(), [](float value)
                                                            { return std::isfinite(value); }))
    {
        return juce::Result::fail("Vocoder residual: synthesis produced an invalid sample count or a non-finite value.");
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    output = std::move(samples);
    return juce::Result::ok();
}
} // namespace sv::synthesis
