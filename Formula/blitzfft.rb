class Blitzfft < Formula
  desc "Native Rust FFT analysis and benchmarking CLI for audio"
  homepage "https://github.com/TheColby/BlitzFFT"
  license "MIT"
  head "https://github.com/TheColby/BlitzFFT.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    assert_match "Supported backends", shell_output("#{bin}/blitzfft --list-backends")
    assert_match "Supported precisions", shell_output("#{bin}/blitzfft --list-precisions")
  end
end
