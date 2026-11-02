// Stage timing of ORB-SLAM3's fisheye stereo front end on real frames:
// extraction (two threads, as Frame.cc does) -> BF kNN(2) Hamming -> ratio test -> TriangulateMatches.
#include "ORBextractor.h"
#include "ORBextractorGPU.h"
#include "CameraModels/KannalaBrandt8.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <cstdio>
using namespace ORB_SLAM3;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) {
    if (argc < 3 || (argc - 1) % 2) { fprintf(stderr, "usage: %s left.png right.png [left2 right2 ...]\n", argv[0]); return 1; }
    const bool gpuExtract = getenv("GPU_EXTRACT") ? atoi(getenv("GPU_EXTRACT")) != 0 : true;
    ORBextractor::SetUseGPU(gpuExtract);
    ORBextractor exL(2000, 1.2f, 8, 20, 7), exR(2000, 1.2f, 8, 20, 7);
    std::vector<float> cam = {607.5108682653754f, 609.9053834748036f, 958.2558093864706f, 539.7975091283472f,
                              -0.03645464928389926f, -0.0008403394026266405f, 0.00017697497379393103f, -0.000382711572282332f};
    KannalaBrandt8 c1(cam), c2(cam); c1.mvLappingArea = {0, 1919}; c2.mvLappingArea = {0, 1919};
    Eigen::Matrix3f Rlr; Rlr << 0.9999581038f, 0.0010089898f, 0.0090979406f, -0.0011175407f, 0.9999281605f, 0.0119341921f, -0.0090852455f, -0.0119438595f, 0.9998873949f;
    Eigen::Vector3f tlr(-0.0889903996f, 0.0011567181f, 0.0023003115f);
    std::vector<int> lap = {0, 1919};
    cv::BFMatcher bf(cv::NORM_HAMMING);
    for (int a = 1; a + 1 < argc; a += 2) {
        cv::Mat L = cv::imread(argv[a], cv::IMREAD_GRAYSCALE), R = cv::imread(argv[a + 1], cv::IMREAD_GRAYSCALE);
        if (L.empty() || R.empty()) { printf("cannot read pair %s %s\n", argv[a], argv[a + 1]); continue; }
        std::vector<cv::KeyPoint> kL, kR; cv::Mat dL, dR; int monoL = 0, monoR = 0;
        auto extract = [&]() {
            std::thread tl([&] { monoL = exL(L, cv::Mat(), kL, dL, lap); });
            std::thread tr([&] { monoR = exR(R, cv::Mat(), kR, dR, lap); });
            tl.join(); tr.join();
        };
        extract(); // warm-up (JIT)
        const int N = 10; double tE = 0, tM = 0, tR = 0, tT = 0, tG = 0; int nRatio = 0, nTri = 0;
        std::vector<std::vector<cv::DMatch>> matches;
        const std::vector<float>& sig2 = exL.GetScaleSigmaSquares();
        for (int i = 0; i < N; ++i) {
            auto t0 = Clock::now(); extract(); auto t1 = Clock::now();
            cv::Mat sdL = dL.rowRange(monoL, dL.rows), sdR = dR.rowRange(monoR, dR.rows);
            matches.clear(); bf.knnMatch(sdL, sdR, matches, 2); auto t2 = Clock::now();
            std::vector<std::vector<cv::DMatch>> gm; auto tg0 = Clock::now(); bool gok = GpuHamming::Knn2(sdL, sdR, gm); auto tg1 = Clock::now(); tG += ms(tg0, tg1);
            if (gok && i == 0) { int same = 0, sameDist = 0; for (size_t m = 0; m < matches.size(); ++m) { if (gm[m][0].trainIdx == matches[m][0].trainIdx && gm[m][0].distance == matches[m][0].distance && gm[m][1].distance == matches[m][1].distance) same++; if (gm[m][0].distance == matches[m][0].distance && gm[m][1].distance == matches[m][1].distance) sameDist++; }
              int gpass = 0; for (auto& v : gm) if (v.size() >= 2 && v[0].distance < v[1].distance * 0.7f) gpass++;
              printf("    GPU knn2: %zu queries, identical (best idx + both dists) %d, identical dists %d, ratio pass GPU %d\n", gm.size(), same, sameDist, gpass); }
            else if (!gok && i == 0) printf("    GPU knn2 FAILED\n");
            std::vector<int> good; good.reserve(matches.size());
            for (size_t m = 0; m < matches.size(); ++m) if (matches[m].size() >= 2 && matches[m][0].distance < matches[m][1].distance * 0.7f) good.push_back((int)m);
            auto t3 = Clock::now(); if (i == 0) printf("    CPU knn2: %d queries %d train, ratio pass %zu\n", sdL.rows, sdR.rows, good.size());
            int tri = 0;
            for (int m : good) {
                const cv::KeyPoint& a1 = kL[matches[m][0].queryIdx + monoL]; const cv::KeyPoint& a2 = kR[matches[m][0].trainIdx + monoR];
                Eigen::Vector3f p; float d = c1.TriangulateMatches(&c2, a1, a2, Rlr, tlr, sig2[a1.octave], sig2[a2.octave], p);
                if (d > 0.0001f) tri++;
            }
            auto t4 = Clock::now();
            tE += ms(t0, t1); tM += ms(t1, t2); tR += ms(t2, t3); tT += ms(t3, t4); nRatio = (int)good.size(); nTri = tri;
        }
        printf("%s | %s  L %zu R %zu (mono %d/%d)\n", argv[a], argv[a + 1], kL.size(), kR.size(), monoL, monoR);
        printf("  extract(2 threads, %s) %.1f ms | BF knn2 %.1f ms | GPU knn2 %.2f ms | ratio %.2f ms (%d pass) | triangulate %.1f ms (%d ok) | sum %.1f ms\n",
               gpuExtract ? "GPU" : "CPU", tE / N, tM / N, tG / N, tR / N, nRatio, tT / N, nTri, (tE + tM + tR + tT) / N);
    }
    return 0;
}
