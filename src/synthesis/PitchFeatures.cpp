#include "PitchFeatures.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <utility>

namespace sv::synthesis
{
namespace
{
constexpr std::size_t featureChannels = 10;
constexpr std::size_t maximumFrames = 4 * 1024 * 1024;
// Both alias groups registered at 0x1000bc6a0 construct the same ten-channel
// frontend; Hibiki Koto uses the first group, while Teto uses the second.
constexpr std::array<std::uint64_t, 24> frontendTypes{
    0x326915e583979c55, 0x41cc3de418da4ec3, 0xeb26790afe5fca94, 0x42950b54a390f7a7, 0x9d63afbd152217d3, 0x13129719647e2ca4, 0x86149d203f186f29, 0x1051cecb9d9960d1, 0x5dc551ef55f9d8b3, 0x7430fc6f1646a1dc, 0xf35e015c07d8d996, 0xc6df474d88658ae2, 0x326912e58397973c, 0x41cc3ee418da5076, 0xeb267c0afe5fcfad, 0x42950c54a390f95a, 0x9d63b0bd15221986, 0x13129a19647e31bd, 0x86149a203f186a10, 0x1051cbcb9d995bb8, 0x5dc552ef55f9da66, 0x7430ff6f1646a6f5, 0xf35e005c07d8d7e3, 0xc6df464d8865892f};

juce::Result failure(const juce::String& reason)
{
    return juce::Result::fail("Pitch features: " + reason);
}

std::uint32_t readWord(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset]) | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

bool isMeanScaleNormalizer(const DnniNode& node)
{
    if (node.type == "cmpu1")
    {
        return true;
    }
    constexpr std::array<std::uint64_t, 12> seeds{0x0dcd59189d5a0f24, 0x59346d79970ca21e, 0xcf3519b773b767bf, 0x562c8e41fb7fbee2, 0xbd6d25457e1ed24e, 0x0123456789abcdef, 0x76543210fedcba98, 0x02468aceeca86420, 0xeca864202468acee, 0x70556f5965766947, 0x5a4d5a4d5a4d5a4d, 0x000000626d6f6379};
    for (auto hash : seeds)
    {
        for (const char character : std::string_view{"cmpu1"})
        {
            hash = (hash ^ static_cast<std::uint8_t>(character)) * 0x100000001b3;
        }
        if (hash == node.typeId)
        {
            return true;
        }
    }
    return false;
}

juce::Result findLanguage(const std::vector<PhoneSet>& phoneSets, std::string_view language, std::size_t& output)
{
    if (language.empty())
    {
        return failure("an explicit phoneme language is required.");
    }
    std::size_t found = phoneSets.size();
    for (std::size_t index = 0; index < phoneSets.size(); ++index)
    {
        if (phoneSets[index].name.find(language) != std::string::npos)
        {
            if (found != phoneSets.size())
            {
                return failure("the language matches multiple phone sets.");
            }
            found = index;
        }
    }
    if (found == phoneSets.size())
    {
        return failure("unknown phoneme language: " + juce::String::fromUTF8(language.data(), static_cast<int>(language.size())));
    }
    output = found;
    return juce::Result::ok();
}
} // namespace

juce::Result PitchFeatures::load(const DnniReader& reader, std::size_t nodeIndex)
{
    const auto& nodes = reader.getNodes();
    if (nodeIndex >= nodes.size() || std::find(frontendTypes.begin(), frontendTypes.end(), nodes[nodeIndex].typeId) == frontendTypes.end())
    {
        return failure("the selected node is not the verified gen5 frontend.");
    }
    const auto& node = nodes[nodeIndex];
    const auto payload = reader.getPayload(nodeIndex);
    if (payload.size() != 8 || node.children.size() != 3 || readWord(payload, 0) != 1)
    {
        return failure("unsupported gen5 frontend configuration.");
    }
    PitchFeatures candidate;
    candidate.frameIntervalSeconds = std::bit_cast<float>(readWord(payload, 4));
    if (!std::isfinite(candidate.frameIntervalSeconds) || candidate.frameIntervalSeconds <= 0.0f)
    {
        return failure("the frame interval must be finite and positive.");
    }

    const auto& languageGroup = nodes[node.children[0]];
    if (languageGroup.type != "cmpg1" || languageGroup.payloadSize != 0 || languageGroup.children.empty())
    {
        return failure("the language phone-set group is invalid.");
    }
    candidate.phoneSets.resize(languageGroup.children.size());
    for (std::size_t index = 0; index < languageGroup.children.size(); ++index)
    {
        if (const auto result = readPhoneSet(reader, languageGroup.children[index], candidate.phoneSets[index]); result.failed())
        {
            return result;
        }
    }
    if (const auto result = readPhoneSet(reader, node.children[1], candidate.unifiedPhoneSet); result.failed())
    {
        return result;
    }
    for (const auto& phoneSet : candidate.phoneSets)
    {
        for (const auto& category : phoneSet.categories)
        {
            if (std::find(candidate.unifiedPhoneSet.classes.begin(), candidate.unifiedPhoneSet.classes.end(), category) == candidate.unifiedPhoneSet.classes.end())
            {
                return failure("a phoneme category is absent from the unified class table.");
            }
        }
        for (const auto& symbol : phoneSet.unifiedSymbols)
        {
            if (!symbol.empty() && std::find(candidate.unifiedPhoneSet.symbols.begin(), candidate.unifiedPhoneSet.symbols.end(), symbol) == candidate.unifiedPhoneSet.symbols.end())
            {
                return failure("a phoneme mapping is absent from the unified symbol table.");
            }
        }
    }

    const auto& normalization = nodes[node.children[2]];
    if (!isMeanScaleNormalizer(normalization) || normalization.payloadSize != 0 || normalization.children.size() != 2)
    {
        return failure("the frontend requires a two-vector cmpu1 normalization.");
    }
    if (const auto result = reader.readFloatVector(normalization.children[0], candidate.means); result.failed())
    {
        return result;
    }
    if (const auto result = reader.readFloatVector(normalization.children[1], candidate.scales); result.failed())
    {
        return result;
    }
    if (candidate.means.size() != featureChannels || candidate.scales.size() != featureChannels)
    {
        return failure("score normalization must contain ten channels.");
    }
    *this = std::move(candidate);
    return juce::Result::ok();
}

float PitchFeatures::getFrameIntervalSeconds() const noexcept
{
    return frameIntervalSeconds;
}

std::size_t PitchFeatures::getPhonemeCategoryCount() const noexcept
{
    return unifiedPhoneSet.classes.size();
}

std::size_t PitchFeatures::getLanguageCount() const noexcept
{
    return phoneSets.size();
}

juce::Result PitchFeatures::encode(std::span<const PitchNote> notes, std::span<const PhonemeDuration> phonemes, PitchFeatureOutput& output) const
{
    if (means.size() != featureChannels || frameIntervalSeconds <= 0.0f)
    {
        return failure("no frontend has been loaded.");
    }
    if (notes.empty() || phonemes.empty() || notes.size() > maximumFrames || phonemes.size() > maximumFrames)
    {
        return failure("the note or phoneme sequence is empty or exceeds the size limit.");
    }
    PitchFeatureOutput candidate;
    candidate.noteFeatures = {notes.size(), featureChannels, {}};
    candidate.noteFeatures.values.resize(notes.size() * featureChannels, 0.0f);
    candidate.noteFrameCounts.reserve(notes.size());
    candidate.phonemeFrameCounts.reserve(phonemes.size());
    candidate.phonemeCategories.reserve(phonemes.size());
    candidate.phonemeLanguages.reserve(phonemes.size());
    candidate.phonemeVowels.reserve(phonemes.size());

    // The reference rounds note boundaries with a double addition but stores
    // the running time as float. Its phoneme boundary path remains all-float.
    float noteSeconds = 0.0f;
    const float inverseInterval = 1.0f / frameIntervalSeconds;
    const double inverseNoteInterval = 1.0 / static_cast<double>(frameIntervalSeconds);
    for (std::size_t index = 0; index < notes.size(); ++index)
    {
        const auto& note = notes[index];
        const double seconds = note.syllable.durationSeconds;
        if (!std::isfinite(seconds) || seconds <= 0.0 || note.tone < 0 || note.tone > 5)
        {
            return failure("notes require positive finite durations and a tone in the range 0 to 5.");
        }
        const double endSeconds = static_cast<double>(noteSeconds) + seconds;
        const double startFrame = std::round(noteSeconds * inverseInterval);
        const double endFrame = std::round(endSeconds * inverseNoteInterval);
        if (!std::isfinite(endFrame) || endFrame > maximumFrames || endFrame < startFrame)
        {
            return failure("note frame boundaries exceed the supported range.");
        }
        candidate.noteFrameCounts.push_back(static_cast<std::size_t>(endFrame - startFrame));
        noteSeconds = static_cast<float>(endSeconds);

        auto* features = candidate.noteFeatures.values.data() + index * featureChannels;
        features[0] = note.isSilence || note.isBreath ? 1.0f : 0.0f;
        features[1] = note.isContinuation ? 1.0f : 0.0f;
        features[2] = static_cast<float>(note.syllable.midiPitch);
        features[3] = static_cast<float>(std::log(std::min(seconds + 0.001, 5.0)));
        features[4] = note.isRap ? 1.0f : 0.0f;
    }

    float phonemeSeconds = 0.0f;
    for (const auto& phoneme : phonemes)
    {
        if (!std::isfinite(phoneme.durationSeconds) || phoneme.durationSeconds <= 0.0 || phoneme.syllableIndex >= notes.size())
        {
            return failure("phonemes require positive finite durations and a valid source note.");
        }
        const float seconds = static_cast<float>(phoneme.durationSeconds);
        const float endSeconds = phonemeSeconds + seconds;
        const float startFrame = std::round(phonemeSeconds / frameIntervalSeconds);
        const float endFrame = std::round(endSeconds / frameIntervalSeconds);
        if (!std::isfinite(endFrame) || endFrame > maximumFrames || endFrame < startFrame)
        {
            return failure("phoneme frame boundaries exceed the supported range.");
        }
        candidate.phonemeFrameCounts.push_back(static_cast<std::size_t>(endFrame - startFrame));
        phonemeSeconds = endSeconds;

        std::size_t language = 0;
        if (const auto result = findLanguage(phoneSets, phoneme.language, language); result.failed())
        {
            return result;
        }
        const auto& phoneSet = phoneSets[language];
        const auto symbol = std::find(phoneSet.symbols.begin(), phoneSet.symbols.end(), phoneme.symbol);
        if (symbol == phoneSet.symbols.end())
        {
            return failure("unknown phoneme '" + juce::String::fromUTF8(phoneme.symbol.c_str()) + "' for " + juce::String::fromUTF8(phoneSet.name.c_str()) + ".");
        }
        const auto sourceIndex = static_cast<std::size_t>(symbol - phoneSet.symbols.begin());
        const auto& category = phoneSet.categories[sourceIndex];
        const auto unifiedSymbol = std::find(unifiedPhoneSet.symbols.begin(), unifiedPhoneSet.symbols.end(), phoneSet.unifiedSymbols[sourceIndex]);
        const auto& unifiedCategory = unifiedPhoneSet.categories[static_cast<std::size_t>(unifiedSymbol - unifiedPhoneSet.symbols.begin())];
        const auto classIndex = std::find(unifiedPhoneSet.classes.begin(), unifiedPhoneSet.classes.end(), unifiedCategory);
        candidate.phonemeCategories.push_back(static_cast<std::size_t>(classIndex - unifiedPhoneSet.classes.begin()));
        candidate.phonemeLanguages.push_back(language);
        candidate.phonemeVowels.push_back(category == "vowel" || category == "diphthong" ? 1 : 0);
    }

    std::size_t noteStartFrame = 0;
    std::size_t phonemeStartFrame = 0;
    std::size_t phonemeIndex = 0;
    for (std::size_t index = 0; index < notes.size(); ++index)
    {
        const auto& note = notes[index];
        auto* features = candidate.noteFeatures.values.data() + index * featureChannels;
        // 1000bca20 chooses the language at one third of the score interval,
        // after phoneme timing; this also covers an onset in the preceding note.
        const auto languageFrame = noteStartFrame + candidate.noteFrameCounts[index] / 3;
        while (phonemeIndex + 1 < phonemes.size() && phonemeStartFrame + candidate.phonemeFrameCounts[phonemeIndex] < languageFrame)
        {
            phonemeStartFrame += candidate.phonemeFrameCounts[phonemeIndex++];
        }
        const auto& language = phoneSets[candidate.phonemeLanguages[phonemeIndex]].name;
        if (note.isRap && note.tone > 0 && language.find("mandarin") != std::string::npos)
        {
            features[static_cast<std::size_t>(note.tone) + 4] = 1.0f;
        }
        // cmpu1 uses a standard-deviation floor, without clipping the features.
        for (std::size_t channel = 0; channel < featureChannels; ++channel)
        {
            features[channel] = (features[channel] - means[channel]) / std::max(scales[channel], 0.00001f);
        }
        noteStartFrame += candidate.noteFrameCounts[index];
    }
    output = std::move(candidate);
    return juce::Result::ok();
}

juce::Result PitchFeatures::denormalizePitch(std::span<const float> normalizedPitch, std::vector<float>& midiPitch) const
{
    if (means.size() != featureChannels || normalizedPitch.empty() || normalizedPitch.size() > maximumFrames)
    {
        return failure("pitch denormalization requires a loaded frontend and a nonempty bounded sequence.");
    }
    std::vector<float> candidate;
    candidate.reserve(normalizedPitch.size());
    for (const float normalized : normalizedPitch)
    {
        // 1000a6bd0 applies the scalar inverse of numeric feature channel 2.
        const float pitch = means[2] + scales[2] * normalized;
        if (!std::isfinite(pitch))
        {
            return failure("the predicted pitch is non-finite.");
        }
        candidate.push_back(pitch);
    }
    midiPitch = std::move(candidate);
    return juce::Result::ok();
}
} // namespace sv::synthesis
