#pragma once

#include "DnniInference.h"
#include "DnniReader.h"
#include "PitchFeatures.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <functional>
#include <span>

namespace sv::synthesis
{
class PitchContext
{
public:
    struct State
    {
        DnniInference::Cache projection;
        DnniInference::Cache feedForward;
        DnniInference::Cache residual;

        [[nodiscard]] std::size_t getBytes() const noexcept;
    };

    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t nodeIndex, std::size_t phonemeCategoryCount, std::size_t languageCount);
    // Recurrent score and phoneme context is expanded onto the prediction grid.
    // The two output branches share the projection; neither consumes the other.
    [[nodiscard]] juce::Result run(const PitchFeatureOutput& features, std::span<const float> speaker, DnniTensor& feedForwardOutput, DnniTensor& residualOutput, State* state = nullptr, const std::function<bool()>& shouldCancel = {}, DnniRunStatistics* statistics = nullptr) const;

private:
    DnniMatrix phonemeEmbedding;
    DnniMatrix languageEmbedding;
    DnniInference noteEncoder;
    DnniInference phonemeEncoder;
    DnniInference projection;
    DnniInference feedForward;
    DnniInference residual;
    bool loaded = false;
};
} // namespace sv::synthesis
