#pragma once

#include "DnniInference.h"
#include "DnniReader.h"
#include "PhoneSet.h"
#include "PhonemeTiming.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sv::synthesis
{
struct PitchNote
{
    TimingSyllable syllable;
    bool isBreath = false;
    bool isSilence = false;
    bool isContinuation = false;
    bool isRap = false;
    int tone = 0;
    float vibratoModulation = 1.0f;
};

struct PitchFeatureOutput
{
    DnniTensor noteFeatures;
    std::vector<std::size_t> noteFrameCounts;
    std::vector<std::size_t> phonemeFrameCounts;
    std::vector<std::size_t> phonemeCategories;
    std::vector<std::size_t> phonemeLanguages;
    std::vector<std::uint8_t> phonemeVowels;
};

// Owns the gen5 score/phoneme frontend and its normalization statistics.
// Encoding allocates and is only suitable for a synthesis worker.
class PitchFeatures
{
public:
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t nodeIndex);
    [[nodiscard]] float getFrameIntervalSeconds() const noexcept;
    [[nodiscard]] std::size_t getPhonemeCategoryCount() const noexcept;
    [[nodiscard]] std::size_t getLanguageCount() const noexcept;
    [[nodiscard]] juce::Result encode(std::span<const PitchNote> notes, std::span<const PhonemeDuration> phonemes, PitchFeatureOutput& output) const;
    [[nodiscard]] juce::Result denormalizePitch(std::span<const float> normalizedPitch, std::vector<float>& midiPitch) const;

private:
    std::vector<PhoneSet> phoneSets;
    PhoneSet unifiedPhoneSet;
    std::vector<float> means;
    std::vector<float> scales;
    float frameIntervalSeconds = 0.0f;
};
} // namespace sv::synthesis
