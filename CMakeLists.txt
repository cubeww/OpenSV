cmake_minimum_required(VERSION 3.25)
project(OpenSV VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/JUCE/CMakeLists.txt")
    message(FATAL_ERROR "JUCE submodule is missing. Run git submodule update --init --recursive. See BUILD_AND_USAGE.md.")
endif()

add_subdirectory(third_party/JUCE EXCLUDE_FROM_ALL)

juce_add_gui_app(OpenSV
    PRODUCT_NAME "OpenSV"
    COMPANY_NAME "OpenSV"
    BUNDLE_ID "org.opensv.studio"
    VERSION "${PROJECT_VERSION}"
)

target_sources(OpenSV PRIVATE
    src/app/Main.cpp
    src/core/ProjectDocument.cpp
    src/core/MidiFile.cpp
    src/audio/PreviewEngine.cpp
    src/audio/PitchAudition.cpp
    src/ui/StudioTheme.cpp
    src/ui/MainComponent.cpp
    src/ui/EditorWorkspace.cpp
    src/ui/LyricsDialog.cpp
    src/ui/PianoRoll.cpp
    src/ui/ArrangementView.cpp
    src/ui/TrackHeader.cpp
    src/ui/ParameterView.cpp
)

target_include_directories(OpenSV PRIVATE src)
target_compile_definitions(OpenSV PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_MODAL_LOOPS_PERMITTED=0
)

target_link_libraries(OpenSV PRIVATE
    OpenSVEngine
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_data_structures
    juce::juce_gui_extra
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags
)

if(MSVC)
    target_compile_options(OpenSV PRIVATE /utf-8 /permissive- /EHsc)
endif()

add_library(OpenSVEngine STATIC
    src/core/Project.cpp
    src/core/ProjectFile.cpp
    src/core/ParameterCurve.cpp
    src/audio/ProjectRenderer.cpp
    src/audio/RenderVisualization.cpp
    src/audio/WaveFile.cpp
    src/synthesis/VoiceSynthesizer.cpp
    src/synthesis/DnniReader.cpp
    src/synthesis/DnniInference.cpp
    src/synthesis/FeatureNormalizer.cpp
    src/synthesis/AcousticFeatures.cpp
    src/synthesis/AcousticModel.cpp
    src/synthesis/NeuralVocoder.cpp
    src/synthesis/PeriodicExcitation.cpp
    src/synthesis/VocoderFilterBank.cpp
    src/synthesis/VocoderResidual.cpp
    src/synthesis/VocoderSpectralHeads.cpp
    src/synthesis/VoiceDatabase.cpp
    src/synthesis/VoiceConfiguration.cpp
    src/synthesis/PhonemeDictionary.cpp
    src/synthesis/PhonemeTiming.cpp
    src/synthesis/PitchFeatures.cpp
    src/synthesis/PitchContext.cpp
    src/synthesis/PitchDecoder.cpp
    src/synthesis/PitchModel.cpp
    src/synthesis/PitchPostprocess.cpp
    src/synthesis/PitchCurve.cpp
    src/synthesis/PhoneSet.cpp
    src/synthesis/LipRadiation.cpp
)
target_include_directories(OpenSVEngine PUBLIC src)
target_compile_definitions(OpenSVEngine PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(OpenSVEngine PRIVATE
    juce::juce_core
    juce::juce_audio_formats
    juce::juce_dsp
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags
)
if(MSVC)
    target_compile_options(OpenSVEngine PRIVATE /utf-8 /permissive- /EHsc)
endif()

option(OPENSV_BUILD_INTERNAL_TOOLS "Build internal voice diagnostics" OFF)
if(OPENSV_BUILD_INTERNAL_TOOLS)
    juce_add_console_app(OpenSVVoice
        PRODUCT_NAME "opensv-voice"
        COMPANY_NAME "OpenSV"
        BUNDLE_ID "org.opensv.voice"
    )
    set_target_properties(OpenSVVoice PROPERTIES OUTPUT_NAME "opensv-voice")
    target_sources(OpenSVVoice PRIVATE tools/voice/Main.cpp)
    target_link_libraries(OpenSVVoice PRIVATE
        OpenSVEngine
        juce::juce_core
        juce::juce_audio_formats
        juce::juce_dsp
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
    )
    if(MSVC)
        target_compile_options(OpenSVVoice PRIVATE /utf-8 /permissive- /EHsc)
    endif()
endif()
