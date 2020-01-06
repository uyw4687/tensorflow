# Experimental posix filesystem plugin.

package(
    licenses = ["notice"],  # Apache 2.0
)

# See also note at `../posix/BUILD`
cc_library(
    name = "windows_filesystem",
    srcs = [
        "windows_filesystem.cc",
    ],
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
    ],
)
