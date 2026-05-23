load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

# Static-link build of mimalloc for use as a Plasma allocator. mimalloc's
# public symbols are already `mi_*`-prefixed, so unlike jemalloc no extra
# symbol-prefix step is needed. MI_OVERRIDE is disabled so the build does
# not export `malloc`/`free` and cannot interfere with the system allocator
# or with the existing jemalloc LD_PRELOAD use in services.py.
cmake(
    name = "libmimalloc_plasma",
    lib_source = ":all",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "MI_BUILD_STATIC": "ON",
        "MI_BUILD_SHARED": "OFF",
        "MI_BUILD_OBJECT": "OFF",
        "MI_BUILD_TESTS": "OFF",
        "MI_OVERRIDE": "OFF",
        "MI_USE_CXX": "OFF",
    },
    out_static_libs = select({
        "@platforms//os:macos": ["libmimalloc.a"],
        "//conditions:default": ["libmimalloc.a"],
    }),
    visibility = ["//visibility:public"],
)
