# Nexus Engine

![C++20](https://img.shields.io/badge/Standard-C%2B%2B20-blue.svg?style=flat\&logo=c%2B%2B)
![Build](https://img.shields.io/badge/Build-CMake-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Nexus Engine** is a modular, high-performance data processing pipeline built in Modern C++ (C++20). It features a custom memory management system and a plugin-based architecture designed for task orchestration.

The current implementation showcases a **from-scratch AES-128 encryption system**, verifying the engine's capability to handle complex mathematical transformations and binary stream manipulation without reliance on external cryptographic libraries.

---

## Key Features

### Cryptography from Scratch

* **AES-128 Core:** Full implementation of the Advanced Encryption Standard, including Key Expansion, SubBytes, ShiftRows, and MixColumns.
* **Galois Field Arithmetic:** Custom implementation of GF(2⁸) multiplication for the MixColumns transformation.
* **PKCS#7 Padding:** Dynamic padding logic to handle files of arbitrary sizes.

### High-Performance Architecture

* **Zero-Copy Data Flow:** Utilizes `std::shared_ptr<RawBuffer>` to pass large binary payloads between nodes without unnecessary allocation or copying.
* **Custom Memory Buffer:** `RawBuffer` class manages raw heap memory with direct pointer arithmetic, bypassing standard container overhead when needed.
* **Type-Safe Pipelines:** Uses `std::variant` and `std::any` to create type-safe yet flexible data packets that flow through the engine.

### Modular Design

* **Node System:** Polymorphic `BaseNode` architecture allows for easy creation of new processing units (e.g., loggers, filters, encryptors).
* **Pipeline Orchestration:** Simple `engine << node1 << node2` syntax for building complex task graphs.

---

## Project Structure

```text
NexusEngine/
├── src/
│   └── main.cpp           # Entry point & integration test
├── include/
│   ├── NexusEngine.hpp    # Core pipeline orchestrator
│   ├── BaseNode.hpp       # Abstract interface for nodes
│   ├── AESNode.hpp        # AES-128 implementation (Encrypt/Decrypt)
│   ├── AESCore.hpp        # S-boxes, Rcon, and Galois math
│   ├── KeyExpansion.hpp   # 128-bit key schedule logic
│   ├── RawBuffer.hpp     # Custom memory management
│   ├── Types.hpp         # Packet and variant definitions
│   └── LoggerNode.hpp    # Debug and audit tools
└── CMakeLists.txt         # Build configuration
```

---

## Getting Started

### Prerequisites

* **C++ Compiler:** GCC 10+, Clang 12+, or MSVC (must support C++20)
* **CMake:** Version 3.20 or higher

### Build Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/NexusEngine.git
cd NexusEngine
```

2. **Generate build files**

```bash
mkdir build && cd build
cmake ..
```

3. **Compile**

```bash
cmake --build .
```

4. **Run**

```bash
./NexusEngine
```

---

## Usage Example

The current `main.cpp` demonstrates the engine by creating a secret file, encrypting it, and then decrypting it back to verify integrity.

```cpp
NexusEngine engine;

// 1. Define key (128-bit)
std::vector<uint8_t> key = { 0x2b, 0x7e, 0x15, 0x16, /* ... */ };

// 2. Setup pipeline
// File -> AES Encrypt -> Log
engine << std::make_unique<AESNode>("Encryptor", key, AESMode::Encrypt)
       << std::make_unique<LoggerNode>("Audit");

// 3. Process data
auto buffer = load_file("secret.txt");
engine(Packet(buffer));
```

### Output

```text
=== Nexus AES-128 System ===

--- Phase 1: Encryption ---
[AES] Encrypted 57 bytes -> 64 bytes (padded).
[IO] Saved 64 bytes to secret_plans.enc

--- Phase 2: Decryption ---
[AES] Decrypted 64 bytes -> 57 bytes (unpadded).
[IO] Saved 57 bytes to secret_plans_decrypted.txt

Decrypted Message:
"The target location is 45.912, -12.001. Attack at dawn."
```

---

## Disclaimer

This AES implementation is created for **educational purposes** to demonstrate understanding of memory safety, bitwise operations, and algorithm design in C++. It is **not** intended for production security use, as it does not implement protections against side-channel attacks (timing analysis, power monitoring, etc.).

---

## License

MIT License. Free to use for educational and personal projects.
