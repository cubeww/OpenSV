#include "AcousticFeatures.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>

namespace sv::synthesis
{
namespace
{
constexpr std::size_t maximumTensorElements = 128 * 1024 * 1024;

std::uint32_t readUint32(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset]) | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

} // namespace

juce::Result AcousticFeatures::load(const DnniReader& reader, std::size_t nodeIndex)
{
    const auto& nodes = reader.getNodes();
    // The v2/v3 factories at 0x1000e0bb0/0x1000e0e20 use the same payload,
    // runtime vtable and phoneme-mapping mode.
    if (nodeIndex >= nodes.size() || (nodes[nodeIndex].type != "_ftmfv2" && nodes[nodeIndex].type != "_ftmfv3") || nodes[nodeIndex].payloadSize != 32 || nodes[nodeIndex].children.size() != 6)
    {
        return juce::Result::fail("Acoustic features require a 32-byte _ftmfv2 or _ftmfv3 with six children.");
    }
    const auto payload = reader.getPayload(nodeIndex);
    AcousticFeatures candidate;
    candidate.config.phonemeChannels = readUint32(payload, 0);
    candidate.config.pitchChannels = readUint32(payload, 4);
    candidate.config.acousticChannels = readUint32(payload, 8);
    candidate.config.frameIntervalSeconds = std::bit_cast<float>(readUint32(payload, 24));
    if (candidate.config.phonemeChannels < 2 || candidate.config.pitchChannels != 1 || candidate.config.acousticChannels != 71 || readUint32(payload, 12) != 2 || readUint32(payload, 16) != 66 || readUint32(payload, 20) != 0 || readUint32(payload, 28) != 0 || !std::isfinite(candidate.config.frameIntervalSeconds) || candidate.config.frameIntervalSeconds <= 0.0f)
    {
        return juce::Result::fail("Unsupported acoustic feature configuration; expected phoneme/duration features, one F0 channel and 71 output channels without DCT postprocessing.");
    }

    const auto& children = nodes[nodeIndex].children;
    const auto& languageGroup = nodes[children[0]];
    if (languageGroup.type != "cmpg1" || languageGroup.payloadSize != 0 || languageGroup.children.empty())
    {
        return juce::Result::fail("Acoustic features have no valid language phone-set group.");
    }
    candidate.phoneSets.resize(languageGroup.children.size());
    for (std::size_t index = 0; index < languageGroup.children.size(); ++index)
    {
        if (const auto result = readPhoneSet(reader, languageGroup.children[index], candidate.phoneSets[index]); result.failed())
        {
            return result;
        }
    }
    if (const auto result = readPhoneSet(reader, children[1], candidate.unifiedPhoneSet); result.failed())
    {
        return result;
    }
    if (candidate.unifiedPhoneSet.symbols.size() + 1 != candidate.config.phonemeChannels)
    {
        return juce::Result::fail("Unified acoustic phoneme count does not match the feature width.");
    }
    const std::set<std::string> unifiedSymbols(candidate.unifiedPhoneSet.symbols.begin(), candidate.unifiedPhoneSet.symbols.end());
    for (const auto& phoneSet : candidate.phoneSets)
    {
        for (std::size_t index = 0; index < phoneSet.symbols.size(); ++index)
        {
            if (!unifiedSymbols.contains(phoneSet.unifiedSymbols[index]))
            {
                return juce::Result::fail("A language phoneme maps to a missing unified acoustic symbol.");
            }
        }
    }
    const auto& dct = nodes[children[2]];
    if (dct.type != "_dctov0" || dct.payloadSize != 0 || dct.children.size() != 2)
    {
        return juce::Result::fail("Acoustic features have an invalid DCT configuration.");
    }
    for (const auto child : dct.children)
    {
        DnniMatrix matrix;
        if (const auto result = reader.readFloatMatrix(child, matrix); result.failed())
        {
            return result;
        }
        if (matrix.rows != 64 || matrix.columns != 64)
        {
            return juce::Result::fail("Acoustic DCT matrices must have 64 rows and columns.");
        }
    }
    if (const auto result = candidate.phonemeNormalization.load(reader, children[3]); result.failed())
    {
        return result;
    }
    if (const auto result = candidate.pitchNormalization.load(reader, children[4]); result.failed())
    {
        return result;
    }
    if (const auto result = candidate.acousticNormalization.load(reader, children[5]); result.failed())
    {
        return result;
    }
    if (candidate.phonemeNormalization.getChannelCount() != candidate.config.phonemeChannels || candidate.pitchNormalization.getChannelCount() != candidate.config.pitchChannels || candidate.acousticNormalization.getChannelCount() != candidate.config.acousticChannels)
    {
        return juce::Result::fail("Acoustic normalization dimensions do not match the feature configuration.");
    }
    *this = std::move(candidate);
    return juce::Result::ok();
}

const AcousticFeatureConfig& AcousticFeatures::getConfig() const noexcept
{
    return config;
}

const std::vector<PhoneSet>& AcousticFeatures::getPhoneSets() const noexcept
{
    return phoneSets;
}

const PhoneSet& AcousticFeatures::getUnifiedPhoneSet() const noexcept
{
    return unifiedPhoneSet;
}

juce::Result AcousticFeatures::findLanguageIndex(std::string_view language, std::size_t& index) const
{
    if (config.phonemeChannels == 0)
    {
        return juce::Result::fail("Acoustic features have not been loaded.");
    }
    if (language.empty())
    {
        return juce::Result::fail("An explicit acoustic phoneme language is required.");
    }
    std::size_t found = phoneSets.size();
    for (std::size_t candidate = 0; candidate < phoneSets.size(); ++candidate)
    {
        if (phoneSets[candidate].name.find(language) != std::string::npos)
        {
            if (found != phoneSets.size())
            {
                return juce::Result::fail("Acoustic phoneme language matches more than one phone set.");
            }
            found = candidate;
        }
    }
    if (found == phoneSets.size())
    {
        return juce::Result::fail("Unknown acoustic phoneme language: " + juce::String::fromUTF8(language.data(), static_cast<int>(language.size())));
    }
    index = found;
    return juce::Result::ok();
}

juce::Result AcousticFeatures::encodePhonemes(std::span<const TimedPhoneme> phonemes, DnniTensor& output) const
{
    if (config.phonemeChannels == 0)
    {
        return juce::Result::fail("Acoustic features have not been loaded.");
    }
    if (phonemes.empty() || phonemes.size() > maximumTensorElements / config.phonemeChannels)
    {
        return juce::Result::fail("Acoustic phoneme sequence is empty or exceeds the supported tensor size.");
    }
    DnniTensor raw{phonemes.size(), config.phonemeChannels, {}};
    raw.values.resize(raw.frames * raw.channels, 0.0f);
    for (std::size_t index = 0; index < phonemes.size(); ++index)
    {
        const auto& phoneme = phonemes[index];
        if (phoneme.frameCount == 0 || phoneme.frameCount > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        {
            return juce::Result::fail("Acoustic phoneme frame counts must be positive int32 values.");
        }
        std::size_t languageIndex = 0;
        if (const auto result = findLanguageIndex(phoneme.language, languageIndex); result.failed())
        {
            return result;
        }
        const auto& phoneSet = phoneSets[languageIndex];
        const auto source = std::find(phoneSet.symbols.begin(), phoneSet.symbols.end(), phoneme.symbol);
        if (source == phoneSet.symbols.end())
        {
            return juce::Result::fail("Unknown acoustic phoneme '" + juce::String::fromUTF8(phoneme.symbol.c_str()) + "' for " + juce::String::fromUTF8(phoneSet.name.c_str()) + ".");
        }
        const auto sourceIndex = static_cast<std::size_t>(source - phoneSet.symbols.begin());
        const auto& unified = phoneSet.unifiedSymbols[sourceIndex];
        const auto target = std::find(unifiedPhoneSet.symbols.begin(), unifiedPhoneSet.symbols.end(), unified);
        const auto targetIndex = static_cast<std::size_t>(target - unifiedPhoneSet.symbols.begin());
        raw.values[index * raw.channels + targetIndex] = 1.0f;
        const float durationSeconds = static_cast<float>(phoneme.frameCount) * config.frameIntervalSeconds;
        raw.values[(index + 1) * raw.channels - 1] = std::log(durationSeconds + 0.05f);
    }
    return phonemeNormalization.normalize(raw, output);
}

juce::Result AcousticFeatures::normalizeLogF0(std::span<const float> logF0, DnniTensor& output) const
{
    if (logF0.empty() || logF0.size() > maximumTensorElements)
    {
        return juce::Result::fail("Acoustic log-F0 sequence is empty or exceeds the supported tensor size.");
    }
    DnniTensor input{logF0.size(), 1, {logF0.begin(), logF0.end()}};
    return pitchNormalization.normalize(input, output);
}

juce::Result AcousticFeatures::decodeAcoustic71(const DnniTensor& normalized, std::span<const float> logF0, DnniTensor& output) const
{
    if (config.acousticChannels == 0)
    {
        return juce::Result::fail("Acoustic features have not been loaded.");
    }
    if (normalized.frames != logF0.size() || normalized.frames > maximumTensorElements / (config.acousticChannels + 1))
    {
        return juce::Result::fail("Acoustic output and log-F0 frame counts do not match or exceed the supported tensor size.");
    }
    DnniTensor decoded;
    if (const auto result = acousticNormalization.denormalize(normalized, decoded); result.failed())
    {
        return result;
    }
    DnniTensor result{decoded.frames, decoded.channels + 1, {}};
    result.values.resize(result.frames * result.channels);
    for (std::size_t frame = 0; frame < decoded.frames; ++frame)
    {
        const float f0Hz = std::exp(logF0[frame]);
        if (!std::isfinite(logF0[frame]) || !std::isfinite(f0Hz) || f0Hz <= 0.0f)
        {
            return juce::Result::fail("Acoustic log-F0 cannot be represented as a finite positive frequency.");
        }
        auto* source = decoded.values.data() + frame * decoded.channels;
        source[0] = source[0] > 0.5f ? 1.0f : 0.0f;
        source[1] = std::clamp(source[1], 0.01f, 3.0f);
        for (std::size_t channel = 66; channel < 71; ++channel)
        {
            source[channel] = std::clamp(source[channel], 0.0f, 1.0f);
        }
        auto* destination = result.values.data() + frame * result.channels;
        destination[0] = source[0];
        destination[1] = f0Hz;
        std::copy(source + 1, source + decoded.channels, destination + 2);
    }
    output = std::move(result);
    return juce::Result::ok();
}
} // namespace sv::synthesis
