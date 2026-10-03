#include "PitchModel.h"

#include "PitchPostprocess.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <string_view>
#include <utility>

namespace sv::synthesis
{
namespace
{
std::atomic<std::uint64_t> nextModelIdentity{1};
constexpr std::array<std::uint64_t, 12> wrapperTypes{0xcce2240f4753c837, 0x0bc3c8e11c368859, 0xb7d76850e579425a, 0xa81b4e0a342431ed, 0xfd1c9123b5ccf269, 0x736a014c0a00406a, 0x933cd542e8c3839b, 0xdb0be7fe929c02e3, 0xbbcb3355f5331ec9, 0xfdf99c04dc496ac2, 0x8cd901ed074da870, 0x3b69b825e5c1bb3c};
constexpr std::array<std::uint64_t, 12> modelTypes{0x5b8562d18c978b91, 0xe781be473240733f, 0xa8ad7b2715fc5e90, 0xef133a930d94f0fb, 0x0461232f81e7260f, 0x6715e41310bb7ce0, 0xbcd2df869cfd154d, 0xa7dc6559d4dcf415, 0xc310c561c14f056f, 0xc58a4d93dd623a98, 0xe32797f2732f379a, 0x1bd8f79c0eaca55e};

juce::Result fail(const juce::String& message)
{
    return juce::Result::fail("Pitch model: " + message);
}

std::uint32_t readWord(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset]) | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

bool sameNote(const PitchNote& first, const PitchNote& second)
{
    return first.syllable.language == second.syllable.language && first.syllable.phonemes == second.syllable.phonemes && first.syllable.durationSeconds == second.syllable.durationSeconds && first.syllable.midiPitch == second.syllable.midiPitch && first.isBreath == second.isBreath && first.isSilence == second.isSilence && first.isContinuation == second.isContinuation && first.isRap == second.isRap && first.tone == second.tone && first.vibratoModulation == second.vibratoModulation;
}

bool samePhoneme(const PhonemeDuration& first, const PhonemeDuration& second)
{
    return first.language == second.language && first.symbol == second.symbol && first.syllableIndex == second.syllableIndex && first.timingIntervalIndex == second.timingIntervalIndex && first.durationSeconds == second.durationSeconds;
}

void smoothControl(std::vector<float>& values)
{
    // The editor smooths expression controls in both directions on the 5 ms grid.
    for (std::size_t frame = 1; frame < values.size(); ++frame)
    {
        values[frame] = values[frame] * 0.95f + values[frame - 1] * 0.050000011920928955f;
    }
    for (std::size_t end = values.size(); end > 1; --end)
    {
        const auto frame = end - 2;
        values[frame] = values[frame] * 0.95f + values[frame + 1] * 0.050000011920928955f;
    }
}

std::uint32_t hashSeedText(std::string_view text)
{
    std::uint32_t hash = 0x811c9dc5;
    for (const unsigned char character : text)
    {
        const auto signedCharacter = std::bit_cast<std::int8_t>(character);
        hash = (hash ^ static_cast<std::uint32_t>(signedCharacter)) * 0x1000193;
    }
    return hash;
}

std::uint32_t phonemeNoiseSeed(std::span<const PhonemeDuration> phonemes, std::size_t index, std::uint32_t globalSeed)
{
    std::string text = "magic";
    if (index > 0)
    {
        text += phonemes[index - 1].symbol;
    }
    text += '_';
    text += phonemes[index].symbol;
    text += '_';
    if (index + 1 < phonemes.size())
    {
        text += phonemes[index + 1].symbol;
    }
    text += '_';
    text += std::to_string(globalSeed);
    const auto seed = hashSeedText(text);
    return seed != 0 ? seed : hashSeedText(" ");
}

class GaussianNoise
{
public:
    explicit GaussianNoise(std::uint32_t seed) : engine(seed) {}

    void reseed(std::uint32_t seed)
    {
        // A boundary resets only MT19937. The next cached Gaussian sample, if
        // present, still comes from the pair generated before that boundary.
        engine.seed(seed);
    }

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

std::size_t PitchModel::State::getBytes() const noexcept
{
    std::size_t bytes = context.getBytes() + notes.capacity() * sizeof(PitchNote) + phonemes.capacity() * sizeof(PhonemeDuration) + (midiPitch.capacity() + vibratoEnvelope.capacity()) * sizeof(float);
    for (const auto& note : notes)
    {
        bytes += note.syllable.language.capacity() + note.syllable.phonemes.capacity() * sizeof(std::string);
        for (const auto& phoneme : note.syllable.phonemes)
        {
            bytes += phoneme.capacity();
        }
    }
    for (const auto& phoneme : phonemes)
    {
        bytes += phoneme.language.capacity() + phoneme.symbol.capacity();
    }
    return bytes;
}

juce::Result PitchModel::load(const DnniReader& reader, std::size_t rootNode)
try
{
    const auto& nodes = reader.getNodes();
    if (rootNode >= nodes.size() || std::find(wrapperTypes.begin(), wrapperTypes.end(), nodes[rootNode].typeId) == wrapperTypes.end() || nodes[rootNode].children.size() != 1)
    {
        return fail("unsupported F0 model wrapper");
    }
    const auto rootPayload = reader.getPayload(rootNode);
    const auto& model = nodes[nodes[rootNode].children[0]];
    if (rootPayload.size() != 16 || std::find(modelTypes.begin(), modelTypes.end(), model.typeId) == modelTypes.end() || model.children.size() != 5)
    {
        return fail("unsupported gen5 model layout");
    }
    PitchModel candidate;
    candidate.expression = std::bit_cast<float>(readWord(rootPayload, 0));
    candidate.expressionStrength = std::bit_cast<float>(readWord(rootPayload, 4));
    candidate.rapExpression = std::bit_cast<float>(readWord(rootPayload, 8));
    if (!std::isfinite(candidate.expression) || !std::isfinite(candidate.expressionStrength) || !std::isfinite(candidate.rapExpression) || candidate.expression < 0.0f || candidate.expressionStrength < 0.0f || candidate.rapExpression < 0.0f)
    {
        return fail("invalid model expression controls");
    }
    if (auto result = candidate.features.load(reader, model.children[0]); result.failed())
    {
        return result;
    }
    if (auto result = reader.readFloatVector(model.children[1], candidate.speaker); result.failed())
    {
        return result;
    }
    if (candidate.speaker.size() != 32 || candidate.features.getFrameIntervalSeconds() != 0.005f)
    {
        return fail("unsupported speaker dimensions or prediction interval");
    }
    if (auto result = candidate.enhancementProjection.load(reader, model.children[2]); result.failed())
    {
        return result;
    }
    if (auto result = candidate.context.load(reader, model.children[3], candidate.features.getPhonemeCategoryCount(), candidate.features.getLanguageCount()); result.failed())
    {
        return result;
    }
    if (auto result = candidate.decoder.load(reader, model.children[4]); result.failed())
    {
        return result;
    }
    candidate.loaded = true;
    candidate.modelIdentity = nextModelIdentity.fetch_add(1, std::memory_order_relaxed);
    *this = std::move(candidate);
    return juce::Result::ok();
}
catch (const std::bad_alloc&)
{
    return fail("insufficient memory to load model parameters");
}

float PitchModel::getFrameIntervalSeconds() const noexcept
{
    return features.getFrameIntervalSeconds();
}

juce::Result PitchModel::run(std::span<const PitchNote> notes, std::span<const PhonemeDuration> phonemes, std::vector<float>& midiPitch, std::uint32_t noiseSeed, State* state, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics, std::span<const float> vibratoEnvelope) const
try
{
    if (!loaded)
    {
        return fail("no model has been loaded");
    }
    if (state != nullptr)
    {
        state->reusedPrediction = false;
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    if (notes.empty() || phonemes.empty())
    {
        return fail("a complete score and phoneme sequence is required");
    }
    if (state != nullptr && state->modelIdentity == modelIdentity && state->noiseSeed == noiseSeed && std::equal(notes.begin(), notes.end(), state->notes.begin(), state->notes.end(), sameNote) && std::equal(phonemes.begin(), phonemes.end(), state->phonemes.begin(), state->phonemes.end(), samePhoneme) && std::equal(vibratoEnvelope.begin(), vibratoEnvelope.end(), state->vibratoEnvelope.begin(), state->vibratoEnvelope.end()))
    {
        midiPitch = state->midiPitch;
        state->reusedPrediction = true;
        return juce::Result::ok();
    }
    int minimumPitch = 128;
    int maximumPitch = 0;
    for (const auto& note : notes)
    {
        if (note.syllable.midiPitch < 0 || note.syllable.midiPitch > 127 || !std::isfinite(note.vibratoModulation))
        {
            return fail("note pitch or vibrato modulation is invalid");
        }
        minimumPitch = std::min(minimumPitch, note.syllable.midiPitch);
        maximumPitch = std::max(maximumPitch, note.syllable.midiPitch);
    }
    // The original wrapper transposes the complete phrase into its trained range,
    // then restores the requested register after prediction.
    int transpose = maximumPitch > 79 ? 79 - maximumPitch : 0;
    if (minimumPitch < 36)
    {
        transpose = 36 - minimumPitch;
    }
    std::vector<PitchNote> modelNotes(notes.begin(), notes.end());
    for (auto& note : modelNotes)
    {
        note.syllable.midiPitch += transpose;
    }
    PitchFeatureOutput encoded;
    if (auto result = features.encode(modelNotes, phonemes, encoded); result.failed())
    {
        return result;
    }
    DnniTensor enhancedSpeaker;
    if (auto result = enhancementProjection.run({1, 1, {1.0f}}, enhancedSpeaker, nullptr, nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (enhancedSpeaker.frames != 1 || enhancedSpeaker.channels != speaker.size() || enhancedSpeaker.values.size() != speaker.size())
    {
        return fail("enhancement projection returned an invalid speaker shape");
    }
    for (std::size_t channel = 0; channel < speaker.size(); ++channel)
    {
        enhancedSpeaker.values[channel] += speaker[channel];
    }
    DnniTensor feedForward;
    DnniTensor residual;
    if (auto result = context.run(encoded, enhancedSpeaker.values, feedForward, residual, state != nullptr ? &state->context : nullptr, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    const auto frames = residual.frames;
    if (frames == 0 || feedForward.frames != frames)
    {
        return fail("the score context has an invalid frame count");
    }
    if (!vibratoEnvelope.empty() && (vibratoEnvelope.size() < frames || !std::all_of(vibratoEnvelope.begin(), vibratoEnvelope.begin() + static_cast<std::ptrdiff_t>(frames), [](float value)
                                                                                     { return std::isfinite(value); })))
    {
        return fail("the vibrato envelope must contain a finite value for each prediction frame");
    }
    std::vector<float> expressiveFrames(frames, expression);
    std::vector<float> vibratoFrames(frames, 0.0f);
    std::vector<float> scorePitch;
    scorePitch.reserve(modelNotes.size());
    std::size_t frame = 0;
    for (std::size_t noteIndex = 0; noteIndex < modelNotes.size(); ++noteIndex)
    {
        const auto& note = modelNotes[noteIndex];
        scorePitch.push_back(static_cast<float>(note.syllable.midiPitch));
        const auto count = encoded.noteFrameCounts[noteIndex];
        if (count > frames - frame)
        {
            return fail("note durations exceed the context frame count");
        }
        const float noteExpression = note.isRap ? expression * rapExpression : expression;
        std::fill_n(expressiveFrames.begin() + static_cast<std::ptrdiff_t>(frame), count, noteExpression);
        for (std::size_t within = 0; within < count; ++within)
        {
            const float envelope = vibratoEnvelope.empty() ? 1.0f : vibratoEnvelope[frame + within];
            vibratoFrames[frame + within] = std::clamp(note.vibratoModulation * envelope, 0.0f, 2.0f) - 1.0f;
        }
        frame += count;
    }
    smoothControl(expressiveFrames);
    DnniTensor controls{frames, 3, std::vector<float>(frames * 3)};
    std::vector<float> noise(frames);
    GaussianNoise noiseSource(123);
    std::size_t nextPhoneme = 0;
    std::size_t nextPhonemeFrame = 0;
    std::size_t nextNote = 0;
    std::size_t nextNoteFrame = 0;
    std::uint32_t activePhonemeSeed = 0;
    for (frame = 0; frame < frames; ++frame)
    {
        bool reseed = false;
        // Each original event stream consumes at most one boundary per frame,
        // including zero-frame intervals. A note boundary emits an event even
        // when its default take ID is zero and leaves the phoneme seed intact.
        if (nextPhoneme < phonemes.size() && frame >= nextPhonemeFrame)
        {
            activePhonemeSeed = phonemeNoiseSeed(phonemes, nextPhoneme, noiseSeed);
            nextPhonemeFrame += encoded.phonemeFrameCounts[nextPhoneme++];
            reseed = true;
        }
        if (nextNote < modelNotes.size() && frame >= nextNoteFrame)
        {
            nextNoteFrame += encoded.noteFrameCounts[nextNote++];
            reseed = true;
        }
        if (reseed)
        {
            noiseSource.reseed(activePhonemeSeed);
        }
        const auto expressive = expressiveFrames[frame];
        float strength = expressionStrength;
        if (expressive > 0.8f)
        {
            strength = static_cast<float>(static_cast<double>(strength) * std::max(0.0, 3.0 - 2.5 * static_cast<double>(expressive)));
        }
        controls.values[frame * 3] = 0.995f;
        controls.values[frame * 3 + 1] = strength;
        controls.values[frame * 3 + 2] = 0.0f;
        noise[frame] = noiseSource.next() * std::min(1.0f, expressive * 2.0f);
    }
    std::vector<std::uint8_t> vowelFrames(frames, 0);
    frame = 0;
    for (std::size_t phone = 0; phone < encoded.phonemeFrameCounts.size(); ++phone)
    {
        const auto count = encoded.phonemeFrameCounts[phone];
        if (count > frames - frame)
        {
            return fail("phoneme durations exceed the context frame count");
        }
        std::fill_n(vowelFrames.begin() + static_cast<std::ptrdiff_t>(frame), count, encoded.phonemeVowels[phone]);
        frame += count;
    }
    PitchDecoderNotes decoderNotes{encoded.noteFrameCounts, scorePitch, vowelFrames, {}, {}, {}, {}};
    DnniTensor normalized;
    if (auto result = decoder.run(residual, feedForward, noise, controls, vibratoFrames, features, decoderNotes, normalized, shouldCancel, statistics); result.failed())
    {
        return result;
    }
    if (normalized.frames != frames || normalized.channels != 1 || normalized.values.size() != frames)
    {
        return fail("pitch decoder returned an invalid output shape");
    }
    std::vector<float> smoothed(frames);
    for (frame = 0; frame < frames; ++frame)
    {
        float value = frame > 1 ? normalized.values[frame - 1] * 0.25f : 0.0f;
        if (frame > 0)
        {
            value += normalized.values[frame] * 0.5f;
        }
        if (frame + 1 < frames)
        {
            value += normalized.values[frame + 1] * 0.25f;
        }
        smoothed[frame] = value;
    }
    std::vector<float> resultPitch;
    if (auto result = features.denormalizePitch(smoothed, resultPitch); result.failed())
    {
        return result;
    }
    if (auto result = applyPitchPostprocess(modelNotes, features.getFrameIntervalSeconds(), resultPitch, shouldCancel); result.failed())
    {
        return result;
    }
    for (auto& pitch : resultPitch)
    {
        pitch -= static_cast<float>(transpose);
        if (!std::isfinite(pitch))
        {
            return fail("the predicted pitch contains a non-finite value");
        }
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    if (state != nullptr)
    {
        state->modelIdentity = 0;
        state->notes.assign(notes.begin(), notes.end());
        state->phonemes.assign(phonemes.begin(), phonemes.end());
        state->midiPitch = resultPitch;
        state->vibratoEnvelope.assign(vibratoEnvelope.begin(), vibratoEnvelope.end());
        state->noiseSeed = noiseSeed;
        state->modelIdentity = modelIdentity;
    }
    midiPitch = std::move(resultPitch);
    return juce::Result::ok();
}
catch (const std::bad_alloc&)
{
    return fail("insufficient memory for pitch prediction");
}
} // namespace sv::synthesis
