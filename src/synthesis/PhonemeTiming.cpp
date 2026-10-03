#include "PhonemeTiming.h"

#include <algorithm>
#include <array>
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
constexpr std::size_t featureChannels = 272;
constexpr std::size_t maximumTensorElements = 64 * 1024 * 1024;

struct TimingPhone
{
    std::size_t syllableIndex = 0;
    std::size_t phonemeIndex = 0;
    std::size_t languageIndex = 0;
    std::size_t unifiedIndex = 0;
    bool isNucleus = false;
};

struct TimingInterval
{
    std::vector<TimingPhone> phones;
    double durationSeconds = 0.0;
    int midiPitch = 60;
};

std::vector<TimingInterval> alignIntervals(std::vector<TimingInterval> intervals)
{
    // gen2a enables moving the complete onset, including semivowels. A unit
    // without a nucleus is left intact, as is the first unit with no predecessor.
    for (std::size_t index = 1; index < intervals.size(); ++index)
    {
        auto& phones = intervals[index].phones;
        const auto nucleus = std::find_if(phones.begin(), phones.end(), [](const auto& phone)
                                          { return phone.isNucleus; });
        if (nucleus != phones.end())
        {
            auto& preceding = intervals[index - 1].phones;
            preceding.insert(preceding.end(), phones.begin(), nucleus);
            phones.erase(phones.begin(), nucleus);
        }
    }

    std::vector<TimingInterval> result;
    result.reserve(intervals.size());
    for (auto& interval : intervals)
    {
        const auto nuclei = static_cast<std::size_t>(std::count_if(interval.phones.begin(), interval.phones.end(), [](const auto& phone)
                                                                   { return phone.isNucleus; }));
        if (nuclei <= 1)
        {
            result.push_back(std::move(interval));
            continue;
        }
        // The reference splits immediately before each subsequent nucleus,
        // retaining intervening consonants in the preceding interval.
        const double seconds = interval.durationSeconds / static_cast<double>(nuclei);
        std::size_t begin = 0;
        bool foundNucleus = false;
        for (std::size_t index = 0; index < interval.phones.size(); ++index)
        {
            if (interval.phones[index].isNucleus)
            {
                if (foundNucleus)
                {
                    result.push_back({{interval.phones.begin() + static_cast<std::ptrdiff_t>(begin), interval.phones.begin() + static_cast<std::ptrdiff_t>(index)}, seconds, interval.midiPitch});
                    begin = index;
                }
                foundNucleus = true;
            }
        }
        result.push_back({{interval.phones.begin() + static_cast<std::ptrdiff_t>(begin), interval.phones.end()}, seconds, interval.midiPitch});
    }
    return result;
}
// The twelve serialized aliases select factory 0x1000e6720 and 0x1000e6580, respectively.
constexpr std::array<std::uint64_t, 12> durationTypes{
    0x4c30eaff390f599a, 0x210ae732bcda3950, 0xedb2643ec31049d7, 0x76b77e5bf8d6500c, 0x8e0f0182b380a860, 0xd4c2d7946f91e167, 0x135ee08897501166, 0x71a4b344df3f7ece, 0xa4f623b00c5a97c0, 0x1fad6bd2dfa1a17f, 0x5d5d600f0e5453e9, 0x1db558504d28a4ad};
constexpr std::array<std::uint64_t, 12> frontendTypes{
    0xff369e74e595e206, 0xea62ca7c31e37ca0, 0x319b1e6e37917bd3, 0x05909feba966bb54, 0x18b5591e22aba550, 0x825a31102572a303, 0xf0c8bdd6e4fe863a, 0x77f5f7f551d96e82, 0xd9177b5063843fb0, 0xa097f32e2e5fe07b, 0xc6ea192cf9c095d1, 0x32870689962bff1d};

juce::Result failure(const juce::String& reason)
{
    return juce::Result::fail("Phoneme timing: " + reason);
}

std::uint32_t readWord(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset]) | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

juce::Result readEmbedding(const DnniReader& reader, std::size_t nodeIndex, std::size_t rows, std::size_t columns, DnniMatrix& matrix)
{
    const auto& node = reader.getNodes()[nodeIndex];
    if (node.type != "modl4" || node.payloadSize != 0 || node.children.size() != 1)
    {
        return failure("an embedding must be an empty modl4 node with one matrix.");
    }
    if (const auto result = reader.readFloatMatrix(node.children.front(), matrix); result.failed())
    {
        return result;
    }
    if (matrix.rows != rows || matrix.columns != columns)
    {
        return failure("an embedding matrix has unsupported dimensions.");
    }
    return juce::Result::ok();
}

void copyEmbedding(const DnniMatrix& matrix, std::size_t column, float* destination)
{
    for (std::size_t row = 0; row < matrix.rows; ++row)
    {
        destination[row] = matrix.values[row * matrix.columns + column];
    }
}

void encodeScalar(float value, std::span<float> destination)
{
    const auto half = destination.size() / 2;
    for (std::size_t index = 0; index < half; ++index)
    {
        const float angle = value * std::exp2(static_cast<float>(index) * -26.575424194335938f / static_cast<float>(destination.size()));
        destination[index] = std::sin(angle);
        destination[index + half] = std::cos(angle);
    }
}
} // namespace

juce::Result quantizePhonemeDurations(std::span<const PhonemeDuration> durations, float frameIntervalSeconds, std::vector<TimedPhoneme>& output)
{
    if (durations.empty() || durations.size() > maximumTensorElements / featureChannels || !std::isfinite(frameIntervalSeconds) || frameIntervalSeconds <= 0.0f)
    {
        return failure("invalid phoneme sequence or acoustic frame interval for time quantization.");
    }
    const double framesPerSecond = 1.0 / static_cast<double>(frameIntervalSeconds);
    double accumulatedSeconds = 0.0;
    std::int32_t previousBoundary = 0;
    std::vector<TimedPhoneme> quantized;
    quantized.reserve(durations.size());
    for (const auto& phoneme : durations)
    {
        if (!std::isfinite(phoneme.durationSeconds) || phoneme.durationSeconds <= 0.0 || phoneme.language.empty() || phoneme.symbol.empty())
        {
            return failure("time quantization requires positive finite durations and explicit phonemes.");
        }
        accumulatedSeconds += phoneme.durationSeconds;
        const double boundary = accumulatedSeconds * framesPerSecond;
        if (!std::isfinite(boundary) || boundary > static_cast<double>(std::numeric_limits<std::int32_t>::max()) || previousBoundary == std::numeric_limits<std::int32_t>::max())
        {
            return failure("quantized phoneme boundaries exceed the supported int32 frame range.");
        }
        const auto nextBoundary = std::max(previousBoundary + 1, static_cast<std::int32_t>(boundary));
        quantized.push_back({phoneme.language, phoneme.symbol, static_cast<std::size_t>(nextBoundary - previousBoundary)});
        previousBoundary = nextBoundary;
    }
    output = std::move(quantized);
    return juce::Result::ok();
}

juce::Result PhonemeTiming::load(const DnniReader& reader, std::size_t rootNode)
{
    const auto& nodes = reader.getNodes();
    if (rootNode >= nodes.size() || std::find(durationTypes.begin(), durationTypes.end(), nodes[rootNode].typeId) == durationTypes.end())
    {
        return failure("the selected node is not the verified gen2a duration model.");
    }
    const auto& children = nodes[rootNode].children;
    const auto payload = reader.getPayload(rootNode);
    if (children.size() != 6 || payload.size() != 8 || readWord(payload, 0) != 32 || readWord(payload, 4) != 16)
    {
        return failure("unsupported gen2a feature layout.");
    }

    const auto& frontend = nodes[children[0]];
    const auto frontendPayload = reader.getPayload(children[0]);
    if (std::find(frontendTypes.begin(), frontendTypes.end(), frontend.typeId) == frontendTypes.end() || frontend.children.size() != 4 || frontendPayload.size() != 8)
    {
        return failure("unsupported duration feature frontend.");
    }
    PhonemeTiming candidate;
    candidate.minimumPitch = std::bit_cast<std::int32_t>(readWord(frontendPayload, 0));
    candidate.maximumPitch = std::bit_cast<std::int32_t>(readWord(frontendPayload, 4));
    if (candidate.minimumPitch < 0 || candidate.maximumPitch > 127 || candidate.minimumPitch >= candidate.maximumPitch)
    {
        return failure("invalid MIDI pitch normalization range.");
    }

    const auto& languageGroup = nodes[frontend.children[0]];
    if (languageGroup.type != "cmpg1" || languageGroup.payloadSize != 0 || languageGroup.children.empty())
    {
        return failure("the duration frontend requires a nonempty language phone-set group.");
    }
    candidate.phoneSets.resize(languageGroup.children.size());
    for (std::size_t index = 0; index < languageGroup.children.size(); ++index)
    {
        if (const auto result = readPhoneSet(reader, languageGroup.children[index], candidate.phoneSets[index]); result.failed())
        {
            return result;
        }
    }
    if (const auto result = readPhoneSet(reader, frontend.children[1], candidate.unifiedPhoneSet); result.failed())
    {
        return result;
    }
    const std::set<std::string> unifiedSymbols(candidate.unifiedPhoneSet.symbols.begin(), candidate.unifiedPhoneSet.symbols.end());
    for (const auto& phoneSet : candidate.phoneSets)
    {
        for (std::size_t index = 0; index < phoneSet.symbols.size(); ++index)
        {
            if (!unifiedSymbols.contains(phoneSet.unifiedSymbols[index]))
            {
                return failure("a language phoneme maps to an unknown unified symbol.");
            }
        }
    }
    if (const auto result = candidate.inputNormalization.load(reader, frontend.children[2]); result.failed())
    {
        return result;
    }
    if (const auto result = candidate.outputNormalization.load(reader, frontend.children[3]); result.failed())
    {
        return result;
    }
    if (candidate.inputNormalization.getChannelCount() != 1 || candidate.outputNormalization.getChannelCount() != 1)
    {
        return failure("duration normalizers must each contain one channel.");
    }
    if (const auto result = readEmbedding(reader, children[2], 32, candidate.phoneSets.size(), candidate.languageEmbedding); result.failed())
    {
        return result;
    }
    if (const auto result = readEmbedding(reader, children[3], 32, candidate.unifiedPhoneSet.symbols.size(), candidate.phonemeEmbedding); result.failed())
    {
        return result;
    }
    if (const auto result = readEmbedding(reader, children[4], 16, 8, candidate.positionEmbedding); result.failed())
    {
        return result;
    }
    if (const auto result = reader.readFloatVector(children[5], candidate.voiceEmbedding); result.failed())
    {
        return result;
    }
    if (candidate.voiceEmbedding.size() != 128)
    {
        return failure("the duration voice embedding must contain 128 channels.");
    }
    if (const auto result = candidate.network.load(reader, children[1]); result.failed())
    {
        return result;
    }
    *this = std::move(candidate);
    return juce::Result::ok();
}

juce::Result PhonemeTiming::predict(std::span<const TimingSyllable> syllables, std::vector<PhonemeDuration>& output) const
{
    if (phoneSets.empty())
    {
        return failure("no duration model has been loaded.");
    }
    if (syllables.empty() || syllables.size() > maximumTensorElements / featureChannels)
    {
        return failure("the phrase is empty or exceeds the tensor size limit.");
    }

    std::size_t phonemeCount = 0;
    std::vector<TimingInterval> intervals;
    intervals.reserve(syllables.size());
    for (std::size_t syllableIndex = 0; syllableIndex < syllables.size(); ++syllableIndex)
    {
        const auto& syllable = syllables[syllableIndex];
        if (syllable.phonemes.empty() || syllable.phonemes.size() > maximumTensorElements / featureChannels - phonemeCount)
        {
            return failure("a syllable is empty or exceeds the tensor size limit.");
        }
        if (!std::isfinite(syllable.durationSeconds) || syllable.durationSeconds <= 0.0 || syllable.durationSeconds > std::numeric_limits<float>::max() || syllable.midiPitch < 0 || syllable.midiPitch > 127)
        {
            return failure("syllable duration must be finite and positive, and pitch must be a MIDI note.");
        }
        phonemeCount += syllable.phonemes.size();
        if (syllable.language.empty())
        {
            return failure("an explicit phoneme language is required.");
        }
        std::size_t languageIndex = phoneSets.size();
        for (std::size_t index = 0; index < phoneSets.size(); ++index)
        {
            if (phoneSets[index].name.find(syllable.language) != std::string::npos)
            {
                if (languageIndex != phoneSets.size())
                {
                    return failure("the language matches more than one phone set.");
                }
                languageIndex = index;
            }
        }
        if (languageIndex == phoneSets.size())
        {
            return failure("unknown phoneme language: " + juce::String::fromUTF8(syllable.language.c_str()));
        }
        const auto& phoneSet = phoneSets[languageIndex];
        TimingInterval interval{{}, syllable.durationSeconds, syllable.midiPitch};
        interval.phones.reserve(syllable.phonemes.size());
        for (std::size_t position = 0; position < syllable.phonemes.size(); ++position)
        {
            const auto& symbol = syllable.phonemes[position];
            const auto source = std::find(phoneSet.symbols.begin(), phoneSet.symbols.end(), symbol);
            if (source == phoneSet.symbols.end())
            {
                return failure("unknown phoneme '" + juce::String::fromUTF8(symbol.c_str()) + "' for " + juce::String::fromUTF8(phoneSet.name.c_str()) + ".");
            }
            const auto sourceIndex = static_cast<std::size_t>(source - phoneSet.symbols.begin());
            const auto& unified = phoneSet.unifiedSymbols[sourceIndex];
            const auto target = std::find(unifiedPhoneSet.symbols.begin(), unifiedPhoneSet.symbols.end(), unified);
            const auto targetIndex = static_cast<std::size_t>(target - unifiedPhoneSet.symbols.begin());
            const auto& category = phoneSet.categories[sourceIndex];
            interval.phones.push_back({syllableIndex, position, languageIndex, targetIndex, category == "vowel" || category == "diphthong"});
        }
        intervals.push_back(std::move(interval));
    }
    intervals = alignIntervals(std::move(intervals));

    DnniTensor rawDuration{intervals.size(), 1, {}};
    rawDuration.values.reserve(intervals.size());
    for (const auto& interval : intervals)
    {
        // The reference stores this offset as a double converted from 0.01f.
        rawDuration.values.push_back(static_cast<float>(std::log(interval.durationSeconds + static_cast<double>(0.01f))));
    }
    DnniTensor normalizedDuration;
    if (const auto result = inputNormalization.normalize(rawDuration, normalizedDuration); result.failed())
    {
        return result;
    }

    DnniTensor features{phonemeCount, featureChannels, {}};
    features.values.resize(phonemeCount * featureChannels);
    std::vector<PhonemeDuration> durations;
    durations.reserve(phonemeCount);
    std::size_t frame = 0;
    for (std::size_t intervalIndex = 0; intervalIndex < intervals.size(); ++intervalIndex)
    {
        const auto& interval = intervals[intervalIndex];
        const float pitch = static_cast<float>(interval.midiPitch - minimumPitch) / static_cast<float>(maximumPitch - minimumPitch);
        for (std::size_t position = 0; position < interval.phones.size(); ++position)
        {
            const auto& phone = interval.phones[position];
            const auto& syllable = syllables[phone.syllableIndex];
            auto* destination = features.values.data() + frame * featureChannels;
            copyEmbedding(phonemeEmbedding, phone.unifiedIndex, destination);
            copyEmbedding(languageEmbedding, phone.languageIndex, destination + 32);
            copyEmbedding(positionEmbedding, std::min(position, std::size_t{7}), destination + 64);
            copyEmbedding(positionEmbedding, std::min(interval.phones.size() - position - 1, std::size_t{7}), destination + 80);
            encodeScalar(normalizedDuration.values[intervalIndex], {destination + 96, 32});
            encodeScalar(pitch, {destination + 128, 16});
            std::copy(voiceEmbedding.begin(), voiceEmbedding.end(), destination + 144);
            durations.push_back({syllable.language, syllable.phonemes[phone.phonemeIndex], phone.syllableIndex, intervalIndex, 0.0});
            ++frame;
        }
    }

    DnniTensor predicted;
    if (const auto result = network.run(features, predicted); result.failed())
    {
        return result;
    }
    if (predicted.frames != phonemeCount || predicted.channels != 1 || predicted.values.size() != phonemeCount)
    {
        return failure("the duration network must return one value per phoneme.");
    }
    DnniTensor logDurations;
    if (const auto result = outputNormalization.denormalize(predicted, logDurations); result.failed())
    {
        return result;
    }

    std::size_t first = 0;
    for (const auto& interval : intervals)
    {
        const auto end = first + interval.phones.size();
        float sum = 0.0f;
        for (auto index = first; index < end; ++index)
        {
            const float value = std::exp(logDurations.values[index]);
            if (!std::isfinite(value) || value <= 0.0f)
            {
                return failure("a predicted duration is not a finite positive float.");
            }
            logDurations.values[index] = value;
            sum += value;
        }
        if (!std::isfinite(sum) || sum <= 0.0f)
        {
            return failure("the predicted timing interval duration sum is invalid.");
        }
        const float reciprocal = 1.0f / sum;
        for (auto index = first; index < end; ++index)
        {
            const float fraction = logDurations.values[index] * reciprocal;
            const float seconds = static_cast<float>(static_cast<double>(fraction) * interval.durationSeconds);
            if (!std::isfinite(seconds) || seconds <= 0.0f)
            {
                return failure("the normalized duration cannot be represented as a positive float.");
            }
            durations[index].durationSeconds = static_cast<double>(seconds);
        }
        first = end;
    }
    output = std::move(durations);
    return juce::Result::ok();
}
} // namespace sv::synthesis
