// CPU-vs-GPU ORBextractor harness (orbslam3_ros2 patch 10 validation).
// Runs the stock CPU path and the GPU path on the same images and reports
// timing, keypoint agreement and descriptor agreement.
#include "ORBextractor.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
#include <string>

using namespace ORB_SLAM3;
using Clock = std::chrono::steady_clock;

static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

static int hamming(const uchar* a, const uchar* b) { int d = 0; for (int i = 0; i < 32; ++i) d += __builtin_popcount(a[i] ^ b[i]); return d; }

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s img.png [img2.png ...]\n", argv[0]); return 1; }
    const int nfeat = 2000, nlevels = 8; const float scale = 1.2f; const int iniTh = 20, minTh = 7;
    ORBextractor cpu(nfeat, scale, nlevels, iniTh, minTh), gpu(nfeat, scale, nlevels, iniTh, minTh);
    std::vector<int> lapping = {0, 0};   // mono ordering only; stereo split is downstream of this test
    for (int a = 1; a < argc; ++a) {
        cv::Mat im = cv::imread(argv[a], cv::IMREAD_GRAYSCALE); if (im.empty()) { printf("  cannot read %s\n", argv[a]); continue; }
        std::vector<cv::KeyPoint> kc, kg; cv::Mat dc, dg;
        // warm-up (GPU: JIT compile + allocations happen here)
        ORBextractor::SetUseGPU(false); cpu(im, cv::Mat(), kc, dc, lapping);
        ORBextractor::SetUseGPU(true);  gpu(im, cv::Mat(), kg, dg, lapping);
        // timing
        const int N = 15; double tc = 0, tg = 0;
        for (int i = 0; i < N; ++i) { auto t0 = Clock::now(); ORBextractor::SetUseGPU(false); cpu(im, cv::Mat(), kc, dc, lapping); tc += ms(t0, Clock::now()); }
        for (int i = 0; i < N; ++i) { auto t0 = Clock::now(); ORBextractor::SetUseGPU(true);  gpu(im, cv::Mat(), kg, dg, lapping); tg += ms(t0, Clock::now()); }
        tc /= N; tg /= N;
        // agreement: for each GPU keypoint, nearest CPU keypoint in the same octave within 1 px
        std::vector<int> perLevelC(nlevels, 0), perLevelG(nlevels, 0);
        for (auto& k : kc) perLevelC[k.octave]++; for (auto& k : kg) perLevelG[k.octave]++;
        int matched = 0, exact = 0; double dAng = 0, dHam = 0; int hamN = 0, ham0 = 0, hamLe2 = 0;
        for (size_t i = 0; i < kg.size(); ++i) {
            int best = -1; float bd = 1e9f;
            for (size_t j = 0; j < kc.size(); ++j) {
                if (kc[j].octave != kg[i].octave) continue;
                float dx = kc[j].pt.x - kg[i].pt.x, dy = kc[j].pt.y - kg[i].pt.y, d = dx * dx + dy * dy;
                if (d < bd) { bd = d; best = (int)j; }
            }
            if (best < 0 || bd > 1.0f) continue;
            matched++; if (bd == 0.f) exact++;
            float da = fabsf(kc[best].angle - kg[i].angle); if (da > 180.f) da = 360.f - da; dAng += da;
            int h = hamming(dc.ptr(best), dg.ptr((int)i)); dHam += h; hamN++; if (h == 0) ham0++; if (h <= 2) hamLe2++;
        }
        printf("  %s  %dx%d\n", argv[a], im.cols, im.rows);
        printf("    time per image: CPU %.1f ms   GPU %.1f ms   (x%.1f)\n", tc, tg, tc / tg);
        printf("    keypoints: CPU %zu  GPU %zu   per level CPU/GPU:", kc.size(), kg.size());
        for (int l = 0; l < nlevels; ++l) printf(" %d/%d", perLevelC[l], perLevelG[l]); printf("\n");
        printf("    GPU keypoints with a CPU keypoint within 1 px (same octave): %d / %zu (%.1f %%), identical position %d\n", matched, kg.size(), 100.0 * matched / std::max<size_t>(kg.size(), 1), exact);
        if (hamN) printf("    on matched keypoints: mean |dAngle| %.2f deg, descriptor Hamming mean %.2f bits, identical %d (%.0f %%), <=2 bits %d (%.0f %%)\n",
                         dAng / hamN, dHam / hamN, ham0, 100.0 * ham0 / hamN, hamLe2, 100.0 * hamLe2 / hamN);
    }
    return 0;
}
