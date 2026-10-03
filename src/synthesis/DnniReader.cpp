#include "DnniReader.h"

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <utility>

namespace sv::synthesis
{
namespace
{
constexpr std::size_t maximumModelBytes = 512 * 1024 * 1024;
constexpr std::size_t maximumNodes = 1000000;
constexpr unsigned maximumDepth = 256;
constexpr std::size_t nodeHeaderBytes = 20;

std::string resolveType(std::uint64_t typeId)
{
    constexpr std::array<std::uint64_t, 12> seeds{0x0dcd59189d5a0f24, 0x59346d79970ca21e, 0xcf3519b773b767bf, 0x562c8e41fb7fbee2, 0xbd6d25457e1ed24e, 0x0123456789abcdef, 0x76543210fedcba98, 0x02468aceeca86420, 0xeca864202468acee, 0x70556f5965766947, 0x5a4d5a4d5a4d5a4d, 0x000000626d6f6379};
    for (const auto* name : {"prim0", "prim1", "prim2", "prim3", "prim4", "prim5", "modm0", "modl0", "modl1", "modl3", "modl4", "modl6", "moda0", "moda1", "moda2", "moda3", "moda4", "moda5", "moda7", "_gnc1v0", "_ncwnv0", "cmpg1", "cmpu0", "cmpu1", "_vocfv1", "_vocfv2", "_ppusv0", "_ppdsv0", "_psv2", "_rldtg0", "_rldms0", "_stbkv1", "_vqctx1", "_didsv0", "_ftmfv2", "_ftmfv3", "_dctov0"})
    {
        for (auto hash : seeds)
        {
            for (const auto* character = name; *character != 0; ++character)
            {
                hash = (hash ^ static_cast<std::uint8_t>(*character)) * 0x100000001b3;
            }
            if (hash == typeId)
            {
                return name;
            }
        }
    }
    return {};
}

std::uint32_t readUint32(const std::uint8_t* bytes)
{
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) | (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

juce::Result malformed(std::size_t offset, const juce::String& reason)
{
    return juce::Result::fail("DNNI offset 0x" + juce::String::toHexString(static_cast<juce::int64>(offset)) + ": " + reason);
}

juce::Result decodeFloats(std::span<const std::uint8_t> bytes, std::size_t count, std::vector<float>& output)
{
    if (count > bytes.size() / sizeof(float) || count * sizeof(float) != bytes.size())
    {
        return juce::Result::fail("DNNI float tensor dimensions do not match its payload.");
    }
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        const auto value = std::bit_cast<float>(readUint32(bytes.data() + index * sizeof(float)));
        if (!std::isfinite(value))
        {
            return juce::Result::fail("DNNI float tensor contains a non-finite value.");
        }
        values[index] = value;
    }
    output = std::move(values);
    return juce::Result::ok();
}

std::int32_t readQuantized(const std::uint8_t* bytes, unsigned bits)
{
    if (bits == 8)
    {
        return std::bit_cast<std::int8_t>(bytes[0]);
    }
    const auto value = static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[0]) | (static_cast<std::uint16_t>(bytes[1]) << 8));
    return std::bit_cast<std::int16_t>(value);
}

juce::Result decodeMatrix(std::span<const std::uint8_t> payload, const std::string& type, DnniMatrix& output)
{
    const bool quantized = type == "prim4" || type == "prim5";
    const bool sparse = type == "prim3" || type == "prim5";
    unsigned bits = 32;
    std::vector<float> scales;
    if (quantized)
    {
        if (payload.size() < 16)
        {
            return juce::Result::fail("DNNI quantization header is truncated.");
        }
        bits = readUint32(payload.data());
        if (bits != 8 && bits != 16)
        {
            return juce::Result::fail("DNNI quantized matrix supports only verified signed 8/16-bit weights.");
        }
        // The second field controls original-engine input quantization, not weight scaling.
        if (readUint32(payload.data() + 8) != 0)
        {
            return juce::Result::fail("DNNI residual quantization is not yet implemented.");
        }
        const auto scaleCount = static_cast<std::size_t>(readUint32(payload.data() + 12));
        if (scaleCount > (payload.size() - 16) / 4)
        {
            return juce::Result::fail("DNNI quantization scale array is truncated.");
        }
        if (auto result = decodeFloats(payload.subspan(16, scaleCount * 4), scaleCount, scales); result.failed())
        {
            return result;
        }
        payload = payload.subspan(16 + scaleCount * 4);
    }
    const std::size_t headerSize = sparse ? 20 : 8;
    if (payload.size() < headerSize)
    {
        return juce::Result::fail("DNNI matrix header is truncated.");
    }
    DnniMatrix matrix;
    matrix.rows = readUint32(payload.data());
    matrix.columns = readUint32(payload.data() + 4);
    const auto elementCount = static_cast<std::uint64_t>(matrix.rows) * matrix.columns;
    if (elementCount > maximumModelBytes / sizeof(float))
    {
        return juce::Result::fail("Decoded DNNI matrix exceeds the 512 MiB limit.");
    }
    if (quantized && scales.size() != matrix.rows)
    {
        return juce::Result::fail("DNNI quantized matrix must have one scale per output row.");
    }
    if (!sparse && !quantized)
    {
        if (auto result = decodeFloats(payload.subspan(8), static_cast<std::size_t>(elementCount), matrix.values); result.failed())
        {
            return result;
        }
        output = std::move(matrix);
        return juce::Result::ok();
    }
    const auto bytesPerElement = static_cast<std::size_t>(bits / 8);
    const float divisor = std::ldexp(1.0f, static_cast<int>(bits) - 1);
    if (!sparse)
    {
        payload = payload.subspan(8);
        if (elementCount * bytesPerElement != payload.size())
        {
            return juce::Result::fail("DNNI quantized matrix dimensions do not match its payload.");
        }
        matrix.values.resize(static_cast<std::size_t>(elementCount));
        for (std::size_t row = 0; row < matrix.rows; ++row)
        {
            for (std::size_t column = 0; column < matrix.columns; ++column)
            {
                const auto integer = readQuantized(payload.data() + (column * matrix.rows + row) * bytesPerElement, bits);
                matrix.values[row * matrix.columns + column] = (static_cast<float>(integer) / divisor) * scales[row];
            }
        }
    }
    else
    {
        const auto blockRows = static_cast<std::size_t>(readUint32(payload.data() + 8));
        const auto blockColumns = static_cast<std::size_t>(readUint32(payload.data() + 12));
        const auto blockCount = static_cast<std::size_t>(readUint32(payload.data() + 16));
        if (blockRows == 0 || blockColumns == 0 || blockRows > matrix.rows || blockColumns > matrix.columns || matrix.rows % blockRows != 0 || matrix.columns % blockColumns != 0)
        {
            return juce::Result::fail("DNNI sparse matrix has unsupported partial or empty block dimensions.");
        }
        const auto blockRowCount = matrix.rows / blockRows;
        const auto indexBytes = static_cast<std::uint64_t>(blockCount) * 2;
        const auto pointerBytes = (static_cast<std::uint64_t>(blockRowCount) + 1) * 4;
        const auto blockElements = blockRows * blockColumns;
        if (blockCount > maximumModelBytes / bytesPerElement / blockElements)
        {
            return juce::Result::fail("DNNI sparse coefficient array exceeds the memory limit.");
        }
        const auto coefficientCount = blockCount * blockElements;
        if (coefficientCount > maximumModelBytes / bytesPerElement || 20 + indexBytes + pointerBytes + coefficientCount * bytesPerElement != payload.size())
        {
            return juce::Result::fail("DNNI sparse matrix arrays do not match its payload.");
        }
        const auto* indices = payload.data() + 20;
        const auto* pointers = indices + static_cast<std::size_t>(indexBytes);
        const auto* coefficients = pointers + static_cast<std::size_t>(pointerBytes);
        if (readUint32(pointers) != 0 || readUint32(pointers + blockRowCount * 4) != blockCount)
        {
            return juce::Result::fail("DNNI sparse row pointers do not span all blocks.");
        }
        matrix.values.resize(static_cast<std::size_t>(elementCount), 0.0f);
        for (std::size_t blockRow = 0; blockRow < blockRowCount; ++blockRow)
        {
            const auto begin = static_cast<std::size_t>(readUint32(pointers + blockRow * 4));
            const auto end = static_cast<std::size_t>(readUint32(pointers + (blockRow + 1) * 4));
            if (begin > end || end > blockCount)
            {
                return juce::Result::fail("DNNI sparse row pointers are out of order or range.");
            }
            for (auto block = begin; block < end; ++block)
            {
                const auto columnBlock = static_cast<std::size_t>(indices[block * 2]) | (static_cast<std::size_t>(indices[block * 2 + 1]) << 8);
                if (columnBlock >= matrix.columns / blockColumns)
                {
                    return juce::Result::fail("DNNI sparse block column is outside the matrix.");
                }
                for (std::size_t column = 0; column < blockColumns; ++column)
                {
                    for (std::size_t row = 0; row < blockRows; ++row)
                    {
                        const auto* coefficient = coefficients + (block * blockRows * blockColumns + column * blockRows + row) * bytesPerElement;
                        const auto outputRow = blockRow * blockRows + row;
                        const auto outputColumn = columnBlock * blockColumns + column;
                        const float value = quantized ? (static_cast<float>(readQuantized(coefficient, bits)) / divisor) * scales[outputRow] : std::bit_cast<float>(readUint32(coefficient));
                        auto& destination = matrix.values[outputRow * matrix.columns + outputColumn];
                        destination += value;
                        if (!std::isfinite(destination))
                        {
                            return juce::Result::fail("DNNI sparse matrix has a non-finite accumulated weight.");
                        }
                    }
                }
            }
        }
    }
    output = std::move(matrix);
    return juce::Result::ok();
}
} // namespace

juce::Result DnniReader::load(const juce::File& file)
{
    if (!file.existsAsFile() || file.getSize() < 8 || file.getSize() > static_cast<juce::int64>(maximumModelBytes))
    {
        return juce::Result::fail("DNNI file is missing or outside the supported size range: " + file.getFullPathName());
    }
    juce::MemoryBlock bytes;
    if (!file.loadFileAsData(bytes))
    {
        return juce::Result::fail("Cannot read DNNI file: " + file.getFullPathName());
    }
    return load(std::move(bytes));
}

juce::Result DnniReader::load(juce::MemoryBlock bytes)
{
    DnniReader replacement;
    replacement.data = std::move(bytes);
    const auto result = replacement.parse();
    if (result.wasOk())
    {
        *this = std::move(replacement);
    }
    return result;
}

std::uint32_t DnniReader::getVersion() const noexcept
{
    return version;
}

const std::vector<DnniNode>& DnniReader::getNodes() const noexcept
{
    return nodes;
}

std::span<const std::uint8_t> DnniReader::getPayload(std::size_t nodeIndex) const
{
    if (nodeIndex >= nodes.size())
    {
        return {};
    }
    const auto& node = nodes[nodeIndex];
    return {static_cast<const std::uint8_t*>(data.getData()) + node.payloadOffset, node.payloadSize};
}

juce::Result DnniReader::parse()
{
    if (data.getSize() < 8 || data.getSize() > maximumModelBytes)
    {
        return malformed(0, "model must contain a header and fit within 512 MiB.");
    }
    const auto* bytes = static_cast<const std::uint8_t*>(data.getData());
    if (readUint32(bytes) != 0x7fca00ff)
    {
        return malformed(0, "unrecognised file signature.");
    }
    version = readUint32(bytes + 4);
    if (version != 1 && version != 2)
    {
        return malformed(4, "unsupported format version " + juce::String(version) + ".");
    }
    std::size_t position = 8;
    std::size_t root = 0;
    const auto result = parseNode(position, 0, root);
    if (result.failed())
    {
        return result;
    }
    if (position != data.getSize())
    {
        return malformed(position, "trailing bytes after the root node.");
    }
    return juce::Result::ok();
}

juce::Result DnniReader::parseNode(std::size_t& position, unsigned depth, std::size_t& nodeIndex)
{
    if (depth > maximumDepth || nodes.size() >= maximumNodes)
    {
        return malformed(position, "node depth or count limit exceeded.");
    }
    if (data.getSize() - position < nodeHeaderBytes)
    {
        return malformed(position, "truncated node header.");
    }
    const auto* bytes = static_cast<const std::uint8_t*>(data.getData()) + position;
    const auto marker = readUint32(bytes);
    if (marker != 0x7fca40ff && marker != 0x7fca41ff)
    {
        return malformed(position, "unrecognised node marker.");
    }
    const auto typeLow = readUint32(bytes + 4);
    const auto typeHigh = readUint32(bytes + 8);
    const auto typeId = static_cast<std::uint64_t>(typeLow) | (static_cast<std::uint64_t>(typeHigh) << 32);
    std::string type;
    bool terminated = false;
    for (std::size_t index = 4; version == 1 && index < 12; ++index)
    {
        const auto character = bytes[index];
        if (character == 0)
        {
            terminated = true;
        }
        else if (terminated || character < 0x20 || character > 0x7e)
        {
            return malformed(position + index, "invalid node type tag.");
        }
        else
        {
            type.push_back(static_cast<char>(character));
        }
    }
    if (version == 2)
    {
        type = resolveType(typeId);
        if (type.empty())
        {
            type = ("0x" + juce::String::toHexString(static_cast<juce::int64>(typeId)).paddedLeft('0', 16)).toStdString();
        }
    }
    if (type.empty())
    {
        return malformed(position + 4, "empty node type tag.");
    }
    auto childCount = readUint32(bytes + 12);
    if (version == 2)
    {
        childCount ^= typeLow ^ typeHigh ^ 0xac5e7bd5;
        if (childCount % 3 != 0)
        {
            return malformed(position + 12, "invalid version 2 child count encoding.");
        }
        childCount /= 3;
    }
    const auto payloadSize = static_cast<std::size_t>(readUint32(bytes + 16));
    if (payloadSize > data.getSize() - position - nodeHeaderBytes)
    {
        return malformed(position + 16, "payload extends past the file.");
    }
    const auto payloadOffset = position + nodeHeaderBytes;
    position = payloadOffset + payloadSize;
    if (childCount > (data.getSize() - position) / nodeHeaderBytes || childCount > maximumNodes - nodes.size())
    {
        return malformed(payloadOffset - 8, "child count exceeds the remaining file.");
    }
    nodeIndex = nodes.size();
    nodes.push_back({marker, typeId, std::move(type), payloadOffset - nodeHeaderBytes, payloadOffset, payloadSize, {}});
    nodes[nodeIndex].children.reserve(childCount);
    for (std::uint32_t child = 0; child < childCount; ++child)
    {
        std::size_t childIndex = 0;
        const auto result = parseNode(position, depth + 1, childIndex);
        if (result.failed())
        {
            return result;
        }
        nodes[nodeIndex].children.push_back(childIndex);
    }
    return juce::Result::ok();
}

juce::Result DnniReader::readFloatVector(std::size_t nodeIndex, std::vector<float>& values) const
{
    if (nodeIndex >= nodes.size() || nodes[nodeIndex].type != "prim1")
    {
        return juce::Result::fail("The requested DNNI node is not a prim1 float vector.");
    }
    const auto payload = getPayload(nodeIndex);
    if (payload.size() < 4)
    {
        return malformed(nodes[nodeIndex].payloadOffset, "truncated vector size.");
    }
    return decodeFloats(payload.subspan(4), readUint32(payload.data()), values);
}

juce::Result DnniReader::readFloatMatrix(std::size_t nodeIndex, DnniMatrix& matrix) const
{
    if (nodeIndex >= nodes.size() || (nodes[nodeIndex].type != "prim0" && nodes[nodeIndex].type != "prim2" && nodes[nodeIndex].type != "prim3" && nodes[nodeIndex].type != "prim4" && nodes[nodeIndex].type != "prim5"))
    {
        return juce::Result::fail("The requested DNNI node is not a supported float matrix.");
    }
    const auto payload = getPayload(nodeIndex);
    return decodeMatrix(payload, nodes[nodeIndex].type, matrix);
}
} // namespace sv::synthesis
