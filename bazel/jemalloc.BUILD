load("@rules_foreign_cc//foreign_cc:configure.bzl", "configure_make")
load("@io_ray//bazel:ray.bzl", "filter_files_with_suffix")

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

configure_make(
    name = "libjemalloc",
    lib_source = ":all",
    linkopts = ["-ldl"],
    copts = ["-fPIC"],
    args = ["-j"],
    out_shared_libs = ["libjemalloc.so"],
    # See https://salsa.debian.org/debian/jemalloc/-/blob/c0a88c37a551be7d12e4863435365c9a6a51525f/debian/rules#L8-23
    # for why we are setting "--with-lg-page" on non x86 hardware here.
    configure_options = ["--disable-static", "--enable-prof", "--enable-prof-libunwind"] +
        select({
            "@platforms//cpu:x86_64": [],
            "//conditions:default": ["--with-lg-page=16"],
        }),
    visibility = ["//visibility:public"],
)


filter_files_with_suffix(
    name = "shared",
    srcs = ["@jemalloc//:libjemalloc"],
    suffix = ".so",
    visibility = ["//visibility:public"],
)

# Second jemalloc build for use as a Plasma allocator (direct link, not
# LD_PRELOAD). All public symbols are prefixed with `ray_je_` so this build
# coexists with the LD_PRELOAD-loaded `libjemalloc` above without symbol
# clashes. Profiling is disabled to avoid baking profile overhead into the
# allocator hot path.
configure_make(
    name = "libjemalloc_plasma",
    lib_source = ":all",
    linkopts = ["-ldl", "-lpthread"],
    copts = ["-fPIC"],
    args = ["-j"],
    out_static_libs = ["libjemalloc.a"],
    configure_options = [
        "--enable-static",
        "--disable-shared",
        "--with-jemalloc-prefix=ray_je_",
    ] + select({
        "@platforms//cpu:x86_64": [],
        "//conditions:default": ["--with-lg-page=16"],
    }),
    visibility = ["//visibility:public"],
)
