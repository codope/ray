load("@rules_foreign_cc//foreign_cc:configure.bzl", "configure_make")
load("@io_ray//bazel:ray.bzl", "filter_files_with_suffix")

# For macOS, use a simpler approach with prebuilt or system jemalloc
# For Linux, use the standard configure_make approach

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

# Linux build
configure_make(
    name = "libjemalloc_linux",
    lib_source = ":all",
    linkopts = ["-ldl"],
    copts = ["-fPIC"],
    args = ["-j"],
    out_shared_libs = ["libjemalloc.so"],
    configure_options = [
        "--disable-static",
        "--enable-prof",
        "--enable-prof-libunwind",
    ] + select({
        "@platforms//cpu:x86_64": [],
        "//conditions:default": ["--with-lg-page=16"],
    }),
    target_compatible_with = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)

# macOS placeholder - we'll use dlmalloc for now on macOS
genrule(
    name = "libjemalloc_macos",
    outs = ["libjemalloc.dylib"],
    cmd = """
        # Placeholder: On macOS, we'll use dlmalloc for now
        # To enable jemalloc on macOS, install via Homebrew: brew install jemalloc
        # Then link to /opt/homebrew/lib/libjemalloc.dylib
        echo "jemalloc not available on macOS in this build" > $@
    """,
    target_compatible_with = ["@platforms//os:macos"],
    visibility = ["//visibility:public"],
)

alias(
    name = "libjemalloc",
    actual = select({
        "@platforms//os:linux": ":libjemalloc_linux",
        "@platforms//os:macos": ":libjemalloc_macos",
        "//conditions:default": ":libjemalloc_linux",
    }),
    visibility = ["//visibility:public"],
)


# Create separate targets for different platforms
filter_files_with_suffix(
    name = "shared_linux",
    srcs = ["@jemalloc//:libjemalloc"],
    suffix = ".so",
    visibility = ["//visibility:public"],
)

filter_files_with_suffix(
    name = "shared_macos",
    srcs = ["@jemalloc//:libjemalloc"],
    suffix = ".dylib",
    visibility = ["//visibility:public"],
)

# Platform-specific shared library selector
alias(
    name = "shared",
    actual = select({
        "@platforms//os:linux": ":shared_linux",
        "@platforms//os:macos": ":shared_macos",
        "//conditions:default": ":shared_linux",
    }),
    visibility = ["//visibility:public"],
)
