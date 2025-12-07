# DynAVX
A quick, fast, and portable way to use SIMD.

## Function
DynAVX is a library that is designed with performance in mind. What it does?
* DynAVX allows you to use scalars, SSE, AVX2, and AVX512 through 1 function
* You do not need to know the kind of SIMD the user has
* It can be used in tandem with a [thread pool](https://github.com/billiuhm/superthread)
* Works with GNUC (g++) or Clang

Since it's designed to be performant and non-conforming to a single CPU's SIMD type; at the beginning it uses `__builtin_cpu_supports("SIMD_TYPE")` and can tell whether a CPU uses SSE, AVX2, or AVX2 at runtime.
If a CPU supports none of the following, or the program is not compiled with g++ or Clang; the library will fallback to scalars without causing errors or crashes.

Then, once a function is called; like `addi`, meaning "add integer", will take 3 vectors' pointers (`vectorA`, `vectorB`, `output`), and then check whether they're the same size (including `output`), then will use a switch and go to the correct SIMD function.
The worst case scenario is 1 `switch` (3 cases + default) and an `if` statement of overhead while detecting SIMD type per-use.

## How to use
DynAVX is simple to install; just put the folder called `simd` into your workspace; if you are updating `simd.cpp` you have to include the flags at the top of the file while turning it into an object file. Otherwise, if you are just using the library; include `simd.h` and make sure you compile `simd.o` with your other files (and put it after the file it is used in).

And example compile command is: `g++ -o main.exe main.cpp extra.cpp simd/simd.o`
Or for updating `simd.cpp`: `g++ -c -o simd/simd.o simd/simd.cpp -msse -msse2 -msse3 -mssse3 -msse4 -msse4 -mavx2 -mavx512f`

To implement SIMD into your code; there are several functions to allow you to do that. Some functions are only available in certain forms.

| Operation | int    | float  | double |
|-----------|--------|--------|--------|
| add       | addi   | addf   | addd   |
| subtract  | subi   | subf   | subd   |
| multiply  | muli   | mulf   | muld   |
| divide    | divi   | divf   | divd   |
| dot       | x      | dotf   | x      |

And for each, you must input 3 vectors; `inputA`, `inputB`, and `output`.
Each vector must be the exact same size; so if `output` isn't the same size as `inputA` or `inputB`, use something along the lines of `output.resize(inputA.size)`.

### Example
```cpp
#include <iostream>
#include <vector>
#include "simd.h"

int main() {
  simd calculator;
  std::vector<int> a = {0, 1, 2, 3, 4, 5};
  std::vector<int> b = {5, 4, 3, 2, 1, 0};
  std::vector<int> c;

  c.resize(a.size());
  addi(a, b, c);

  for (int element : c) {
    std::cout << element << " "; 
  }

  return;
}
```

## To be fixed
* Multiplying integers with SSE requires SSE 4.1
* Dividing integers has no instruction in SSE/AVX so it uses scalars
* Support for MSVC
* No guardrails for division by 0
