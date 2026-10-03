#include "VoiceSynthesizer.h"

#include "AcousticFeatures.h"
#include "DnniReader.h"
#include "VoiceConfiguration.h"
#include "VoiceDatabase.h"

#include <algorithm>
#include <cstddef>
#include <new>
#include <optional>
#include <utility>

namespace sv::synthesis
{
namespace
{
juce::Result readModel(VoiceDatabase& database, const ModelReference& reference, const char* architecture, const char* role, DnniReader& reader)
{
    const auto context = juce::String("Voice synthesizer ") + role + " model: ";
    if (reference.architecture != architecture)
    {
        return juce::Result::fail(context + "unsupported architecture '" + juce::String::fromUTF8(reference.architecture.c_str()) + "', expected '" + architecture + "'");
    }
    const auto& entries = database.getEntries();
    const auto entry = std::find_if(entries.begin(), entries.end(), [&reference](const auto& candidate)
                                    { return candidate.key == reference.key; });
    if (entry == entries.end())
    {
        return juce::Result::fail(context + "the configured NOFS entry is missing");
    }
    juce::MemoryBlock bytes;
    if (const auto result = database.readEntry(*entry, bytes); result.failed())
    {
        return juce::Result::fail(context + result.getErrorMessage());
    }
    if (const auto result = reader.load(std::move(bytes)); result.failed())
    {
        return juce::Result::fail(context + result.getErrorMessage());
    }
    return juce::Result::ok();
}

juce::Result readFrameInterval(const DnniReader& reader, float& frameIntervalSeconds)
{
    std::optional<std::size_t> featureNode;
    const auto& nodes = reader.getNodes();
    for (std::size_t index = 0; index < nodes.size(); ++index)
    {
        if (nodes[index].type != "_ftmfv2" && nodes[index].type != "_ftmfv3")
        {
            continue;
        }
        if (featureNode.has_value())
        {
            return juce::Result::fail("Voice synthesizer: acoustic model contains multiple feature configurations");
        }
        featureNode = index;
    }
    if (!featureNode.has_value())
    {
        return juce::Result::fail("Voice synthesizer: acoustic model has no supported _ftmfv2/v3 feature configuration");
    }
    AcousticFeatures features;
    if (const auto result = features.load(reader, *featureNode); result.failed())
    {
        return result;
    }
    frameIntervalSeconds = features.getConfig().frameIntervalSeconds;
    return juce::Result::ok();
}
} // namespace

std::size_t VoiceSynthesizer::State::getBytes() const noexcept
{
    return pitch.getBytes() + acoustic.getBytes() + vocoder.getBytes();
}

juce::Result VoiceSynthesizer::load(const juce::File& voice)
try
{
    VoiceDatabase database;
    if (const auto result = database.open(voice); result.failed())
    {
        return result;
    }
    VoiceConfiguration configuration;
    if (const auto result = configuration.load(database); result.failed())
    {
        return result;
    }
    VoiceSynthesizer candidate;
    for (const auto& language : juce::StringArray::fromTokens(database.getMetadata().properties[".feature_rap_languages"], false))
    {
        candidate.rapLanguages.push_back(language.toStdString());
    }
    {
        const auto* entry = database.findEntry("f0model-dds");
        if (entry == nullptr)
        {
            return juce::Result::fail("Voice synthesizer: the voice has no supported f0model-dds pitch model");
        }
        juce::MemoryBlock bytes;
        if (const auto result = database.readEntry(*entry, bytes); result.failed())
        {
            return result;
        }
        DnniReader reader;
        if (const auto result = reader.load(std::move(bytes)); result.failed())
        {
            return result;
        }
        if (const auto result = candidate.pitch.load(reader); result.failed())
        {
            return juce::Result::fail("Voice synthesizer pitch model: " + result.getErrorMessage());
        }
    }
    {
        DnniReader reader;
        if (const auto result = readModel(database, configuration.getDurationModel(), "gen2a", "duration", reader); result.failed())
        {
            return result;
        }
        if (const auto result = candidate.timing.load(reader); result.failed())
        {
            return result;
        }
    }
    {
        DnniReader reader;
        if (const auto result = readModel(database, configuration.getAcousticModel(), "dds", "acoustic", reader); result.failed())
        {
            return result;
        }
        if (const auto result = candidate.acoustic.load(reader); result.failed())
        {
            return result;
        }
        if (const auto result = readFrameInterval(reader, candidate.frameIntervalSeconds); result.failed())
        {
            return result;
        }
    }
    {
        DnniReader reader;
        if (const auto result = readModel(database, configuration.getVocoderModel(), "nhv", "vocoder", reader); result.failed())
        {
            return result;
        }
        if (const auto result = candidate.vocoder.load(reader); result.failed())
        {
            return result;
        }
    }
    *this = std::move(candidate);
    return juce::Result::ok();
}
catch (const std::bad_alloc&)
{
    return juce::Result::fail("Voice synthesizer: insufficient memory to load the voice");
}

float VoiceSynthesizer::getFrameIntervalSeconds() const noexcept
{
    return frameIntervalSeconds;
}

float VoiceSynthesizer::getPitchFrameIntervalSeconds() const noexcept
{
    return pitch.getFrameIntervalSeconds();
}

juce::Result VoiceSynthesizer::predictPitch(std::span<const PitchNote> notes, std::vector<float>& midiPitch, const std::function<bool()>& shouldCancel, SynthesisStatistics* statistics, State* state, std::span<const float> vibratoEnvelope) const
try
{
    if (frameIntervalSeconds == 0.0f || notes.empty())
    {
        return juce::Result::fail("Pitch prediction requires a loaded voice and score notes.");
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Pitch prediction was cancelled.");
    }
    const double started = juce::Time::getMillisecondCounterHiRes();
    DnniRunStatistics networkStatistics;
    PitchNote leading;
    leading.syllable = {notes.front().syllable.language, {"sil"}, pitchContextSeconds, notes.front().syllable.midiPitch};
    leading.isSilence = true;
    PitchNote trailing;
    trailing.syllable = {notes.back().syllable.language, {"sil"}, pitchContextSeconds, notes.back().syllable.midiPitch};
    trailing.isSilence = true;
    std::vector<PitchNote> supportedNotes;
    supportedNotes.reserve(notes.size() + 2);
    supportedNotes.push_back(std::move(leading));
    supportedNotes.insert(supportedNotes.end(), notes.begin(), notes.end());
    supportedNotes.push_back(std::move(trailing));
    std::vector<TimingSyllable> syllables;
    syllables.reserve(supportedNotes.size());
    for (auto& note : supportedNotes)
    {
        note.isRap = note.isRap && std::find(rapLanguages.begin(), rapLanguages.end(), note.syllable.language) != rapLanguages.end();
        syllables.push_back(note.syllable);
    }
    // The original F0 task predicts durations in its own half-second context;
    // acoustic timing has different boundary intervals and cannot be reused here.
    std::vector<PhonemeDuration> phonemes;
    if (const auto result = timing.predict(syllables, phonemes); result.failed())
    {
        return result;
    }
    const auto result = pitch.run(supportedNotes, phonemes, midiPitch, 0, state != nullptr ? &state->pitch : nullptr, shouldCancel, &networkStatistics, vibratoEnvelope);
    if (statistics != nullptr)
    {
        statistics->pitchMilliseconds += juce::Time::getMillisecondCounterHiRes() - started;
        statistics->pitchFrames += result.wasOk() && (state == nullptr || !state->pitch.reusedPrediction) ? midiPitch.size() : 0;
        statistics->dnniComputedFrames += networkStatistics.computedFrames;
        statistics->dnniReusedFrames += networkStatistics.reusedFrames;
        statistics->dnniContextFrames += networkStatistics.contextFrames;
    }
    return result;
}
catch (const std::bad_alloc&)
{
    return juce::Result::fail("Voice synthesizer: insufficient memory to predict pitch");
}

juce::Result VoiceSynthesizer::predict(std::span<const TimingSyllable> syllables, std::vector<PhonemeDuration>& output) const
try
{
    if (frameIntervalSeconds == 0.0f)
    {
        return juce::Result::fail("Voice synthesizer: no voice has been loaded");
    }
    return timing.predict(syllables, output);
}
catch (const std::bad_alloc&)
{
    return juce::Result::fail("Voice synthesizer: insufficient memory to predict phoneme timing");
}

juce::Result VoiceSynthesizer::render(std::span<const TimedPhoneme> phonemes, std::span<const float> logF0, NeuralVocoderOutput& output, std::uint32_t seed, const std::function<bool()>& shouldCancel, SynthesisStatistics* statistics, State* state) const
try
{
    DnniRunStatistics networkStatistics;
    const juce::ScopeGuard finishStatistics([statistics, &networkStatistics]
                                            {
        if (statistics != nullptr)
        {
            statistics->dnniComputedFrames += networkStatistics.computedFrames;
            statistics->dnniReusedFrames += networkStatistics.reusedFrames;
            statistics->dnniContextFrames += networkStatistics.contextFrames;
        } });
    if (frameIntervalSeconds == 0.0f)
    {
        return juce::Result::fail("Voice synthesizer: no voice has been loaded");
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    DnniTensor frames;
    const double acousticStarted = juce::Time::getMillisecondCounterHiRes();
    const auto acousticResult = acoustic.run(phonemes, logF0, frames, seed, state != nullptr ? &state->acoustic : nullptr, shouldCancel, &networkStatistics);
    if (statistics != nullptr)
    {
        statistics->acousticMilliseconds += juce::Time::getMillisecondCounterHiRes() - acousticStarted;
        statistics->acousticFrames += acousticResult.wasOk() ? frames.frames : 0;
    }
    if (acousticResult.failed())
    {
        return acousticResult;
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    NeuralVocoderOutput candidate;
    if (const auto result = vocoder.run(frames, candidate, {}, statistics, state != nullptr ? &state->vocoder : nullptr, shouldCancel, &networkStatistics); result.failed())
    {
        return result;
    }
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    output = std::move(candidate);
    return juce::Result::ok();
}
catch (const std::bad_alloc&)
{
    return juce::Result::fail("Voice synthesizer: insufficient memory to render the phrase");
}
} // namespace sv::synthesis
