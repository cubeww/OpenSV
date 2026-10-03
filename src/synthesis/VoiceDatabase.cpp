#include "VoiceDatabase.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>

namespace sv::synthesis
{
namespace
{
constexpr std::uint32_t nofsMagic = 0xf580;
constexpr std::uint32_t supportedVersion = 10;
constexpr std::uint16_t indexBlockType = 0x1000;
constexpr std::uint16_t valueBlockType = 1;
constexpr std::uint64_t fileHeaderSize = 16;
constexpr std::uint32_t maximumEntrySize = 512 * 1024 * 1024;
constexpr std::uint32_t maximumMetadataSize = 1024 * 1024;
constexpr std::uint64_t maximumTotalKeySize = 16 * 1024 * 1024;
constexpr std::uint64_t maximumTotalMetadataSize = 16 * 1024 * 1024;
constexpr std::size_t maximumEntryCount = 100000;

std::uint16_t readUint16(const void* data)
{
    return juce::ByteOrder::littleEndianShort(data);
}

std::uint32_t readUint32(const void* data)
{
    return juce::ByteOrder::littleEndianInt(data);
}

std::uint64_t readUint64(const void* data)
{
    return juce::ByteOrder::littleEndianInt64(data);
}

bool readAt(juce::FileInputStream& stream, std::uint64_t offset, void* destination, int size)
{
    return offset <= static_cast<std::uint64_t>(std::numeric_limits<juce::int64>::max()) && stream.setPosition(static_cast<juce::int64>(offset)) && stream.read(destination, size) == size;
}

juce::Result invalid(const juce::String& detail)
{
    return juce::Result::fail("NOFS: " + detail);
}

bool isText(const juce::MemoryBlock& block, bool allowWhitespace)
{
    if (block.isEmpty())
    {
        return true;
    }
    const auto* data = static_cast<const char*>(block.getData());
    if (block.getSize() > static_cast<std::size_t>(std::numeric_limits<int>::max()) || !juce::CharPointer_UTF8::isValidString(data, static_cast<int>(block.getSize())))
    {
        return false;
    }
    return std::none_of(data, data + block.getSize(), [allowWhitespace](char value)
                        {
        const auto byte = static_cast<unsigned char>(value);
        return byte == 0 || byte == 0x7f || (byte < 0x20 && !(allowWhitespace && (byte == '\n' || byte == '\r' || byte == '\t'))); });
}

juce::String textOf(const juce::MemoryBlock& block)
{
    return block.isEmpty() ? juce::String() : juce::String::fromUTF8(static_cast<const char*>(block.getData()), static_cast<int>(block.getSize()));
}
} // namespace

juce::Result VoiceDatabase::open(const juce::File& newFile)
{
    file = juce::File{};
    stream.reset();
    entries.clear();
    metadata = {};

    auto candidate = newFile.createInputStream();
    if (candidate == nullptr)
    {
        return invalid("cannot open " + newFile.getFullPathName());
    }
    const auto signedSize = candidate->getTotalLength();
    if (signedSize < static_cast<juce::int64>(fileHeaderSize + 10))
    {
        return invalid("truncated file header");
    }
    const auto fileSize = static_cast<std::uint64_t>(signedSize);
    std::array<std::uint8_t, 16> header{};
    if (!readAt(*candidate, 0, header.data(), static_cast<int>(header.size())))
    {
        return invalid("cannot read file header");
    }
    if (readUint32(header.data()) != nofsMagic)
    {
        return invalid("unrecognised file signature");
    }
    if (readUint32(header.data() + 4) != supportedVersion)
    {
        return invalid("unsupported container version");
    }
    if (readUint64(header.data() + 8) != fileSize)
    {
        return invalid("declared file length does not match the file");
    }

    std::array<std::uint8_t, 8> blockHeader{};
    if (!readAt(*candidate, fileHeaderSize, blockHeader.data(), static_cast<int>(blockHeader.size())))
    {
        return invalid("cannot read index block");
    }
    const auto indexSize = readUint32(blockHeader.data());
    if (indexSize < 10 || indexSize > fileSize - fileHeaderSize || readUint16(blockHeader.data() + 4) != indexBlockType)
    {
        return invalid("invalid or unsupported index block");
    }
    std::array<std::uint8_t, 4> lengthBytes{};
    if (!readAt(*candidate, fileHeaderSize + indexSize - 4, lengthBytes.data(), static_cast<int>(lengthBytes.size())) || readUint32(lengthBytes.data()) != indexSize)
    {
        return invalid("index block trailer mismatch");
    }

    // The index hash algorithm is not required for a checked sequential walk.
    // Both observed files contain contiguous value blocks through the exact EOF.
    std::vector<VoiceEntry> parsed;
    std::unordered_set<std::string> keys;
    std::uint64_t totalKeySize = 0;
    auto offset = fileHeaderSize + indexSize;
    while (offset < fileSize)
    {
        if (parsed.size() >= maximumEntryCount || fileSize - offset < 16 || !readAt(*candidate, offset, blockHeader.data(), static_cast<int>(blockHeader.size())))
        {
            return invalid("too many entries or truncated value block header");
        }
        const auto recordSize = readUint32(blockHeader.data());
        const auto recordType = readUint16(blockHeader.data() + 4);
        const auto keySize = readUint16(blockHeader.data() + 6);
        if (recordType != valueBlockType)
        {
            return invalid("unsupported value block type at " + juce::String(static_cast<juce::int64>(offset)));
        }
        if (keySize == 0 || recordSize < 16u + keySize || recordSize > fileSize - offset)
        {
            return invalid("invalid value block size");
        }
        totalKeySize += keySize;
        if (totalKeySize > maximumTotalKeySize)
        {
            return invalid("entry keys exceed the 16 MiB read limit");
        }

        VoiceEntry entry;
        entry.key.setSize(keySize);
        if (!readAt(*candidate, offset + 8, entry.key.getData(), keySize))
        {
            return invalid("cannot read entry key");
        }
        const std::string rawKey(static_cast<const char*>(entry.key.getData()), entry.key.getSize());
        if (!keys.insert(rawKey).second)
        {
            return invalid("duplicate entry key");
        }
        if (isText(entry.key, false))
        {
            entry.name = textOf(entry.key);
        }
        if (!readAt(*candidate, offset + 8 + keySize, lengthBytes.data(), static_cast<int>(lengthBytes.size())))
        {
            return invalid("cannot read entry value length");
        }
        entry.valueSize = readUint32(lengthBytes.data());
        entry.valueOffset = offset + 12 + keySize;
        if (static_cast<std::uint64_t>(entry.valueSize) + keySize + 16 != recordSize)
        {
            return invalid("value length does not match its block");
        }
        if (!readAt(*candidate, offset + recordSize - 4, lengthBytes.data(), static_cast<int>(lengthBytes.size())) || readUint32(lengthBytes.data()) != recordSize)
        {
            return invalid("value block trailer mismatch");
        }
        parsed.push_back(std::move(entry));
        offset += recordSize;
    }
    if (parsed.empty())
    {
        return invalid("container has no entries");
    }

    file = newFile;
    stream = std::move(candidate);
    entries = std::move(parsed);
    auto result = readMetadata();
    if (result.failed())
    {
        file = juce::File{};
        stream.reset();
        entries.clear();
        metadata = {};
    }
    return result;
}

const std::vector<VoiceEntry>& VoiceDatabase::getEntries() const noexcept
{
    return entries;
}

const VoiceMetadata& VoiceDatabase::getMetadata() const noexcept
{
    return metadata;
}

const juce::File& VoiceDatabase::getFile() const noexcept
{
    return file;
}

const VoiceEntry* VoiceDatabase::findEntry(juce::StringRef name) const
{
    const auto found = std::find_if(entries.begin(), entries.end(), [name](const VoiceEntry& entry)
                                    { return entry.name.isNotEmpty() && entry.name == name; });
    return found == entries.end() ? nullptr : &*found;
}

juce::Result VoiceDatabase::readEntry(const VoiceEntry& entry, juce::MemoryBlock& destination)
{
    destination.reset();
    if (stream == nullptr)
    {
        return invalid("no database is open");
    }
    const auto found = std::find_if(entries.begin(), entries.end(), [&entry](const VoiceEntry& existing)
                                    { return existing.key == entry.key && existing.valueOffset == entry.valueOffset && existing.valueSize == entry.valueSize; });
    if (found == entries.end())
    {
        return invalid("entry does not belong to the open database");
    }
    if (entry.valueSize > maximumEntrySize)
    {
        return invalid("entry exceeds the 512 MiB read limit");
    }
    if (entry.valueSize == 0)
    {
        return juce::Result::ok();
    }
    destination.setSize(entry.valueSize);
    if (!readAt(*stream, entry.valueOffset, destination.getData(), static_cast<int>(entry.valueSize)))
    {
        destination.reset();
        return invalid("cannot read entry value");
    }
    return juce::Result::ok();
}

juce::Result VoiceDatabase::readMetadata()
{
    std::uint64_t totalMetadataSize = 0;
    for (const auto& entry : entries)
    {
        if (!entry.name.startsWithChar('.'))
        {
            continue;
        }
        if (entry.valueSize > maximumMetadataSize)
        {
            return invalid("metadata entry is too large: " + entry.name);
        }
        totalMetadataSize += entry.valueSize;
        if (totalMetadataSize > maximumTotalMetadataSize)
        {
            return invalid("metadata exceeds the 16 MiB read limit");
        }
        juce::MemoryBlock value;
        if (auto result = readEntry(entry, value); result.failed())
        {
            return result;
        }
        if (!isText(value, true))
        {
            return invalid("metadata is not valid text: " + entry.name);
        }
        metadata.properties.set(entry.name, textOf(value));
    }
    metadata.name = metadata.properties[".name"];
    metadata.vendor = metadata.properties[".vendor"];
    metadata.language = metadata.properties[".language"];
    metadata.phoneset = metadata.properties[".phoneset"];
    metadata.type = metadata.properties[".type"];
    metadata.languages.addTokens(metadata.properties[".multi"], false);
    metadata.timbreStyles.addTokens(metadata.properties[".timbre_styles"], false);
    const auto version = metadata.properties[".version"];
    if (version.isNotEmpty())
    {
        if (!version.containsOnly("0123456789") || version.length() > 9)
        {
            return invalid("invalid database version metadata");
        }
        metadata.version = version.getIntValue();
    }
    return juce::Result::ok();
}
} // namespace sv::synthesis
