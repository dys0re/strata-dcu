#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace strata::spec {

// DeepInfer-style acceptance/latency selection for one sequence, using observed
// acceptance rather than a model-specific confidence head.
// Decisions use COMPLETED rounds only:
// no current draft token or its confidence can influence its own verification.
// An observed cost pairs a window's verifier/commit with the draft that BUILT
// that window, not the possibly different draft length prepared afterwards.
class MtpBudget {
public:
    static constexpr int kMaxT = 8;
    explicit MtpBudget(int cap) : cap_(std::clamp(cap, 1, kMaxT)) {}

    void begin_request() {
        // Keep the hardware cost curve, but do not price a new language/task
        // solely from the previous request's acceptance distribution.
        for (int j = 1; j <= cap_; ++j) {
            prefix_ok_[j] *= .1;
            prefix_n_[j] *= .1;
            auto& s = sizes_[j];
            if (s.samples > 0) {
                const double prior = cost_ms(j);
                s.cost.fill(0);
                s.cost[0] = prior;
                s.samples = 1;
                s.next = 1;
            }
        }
        request_seen_.fill(false);
        request_trials_.fill(0);
        request_probes_ = true;
    }

    int choose(int cap) const {
        cap = std::clamp(cap, 1, cap_);
        // Measure each size; its first observation may include graph capture.
        int probe = 0;
        for (int t = 1; t <= cap; ++t)
            if (sizes_[t].samples < 3 && (!probe || sizes_[t].samples < sizes_[probe].samples ||
                (sizes_[t].samples == sizes_[probe].samples && request_trials_[t] < request_trials_[probe]))) probe = t;
        if (probe) return probe;
        if (request_probes_)
            for (int t = 2; t <= cap; ++t)
                if (!request_seen_[t]) return t;
        for (int t = 1; t <= cap; ++t)
            if (rounds_ - sizes_[t].last > 128) return t;
        int best = 1;
        double score = expected_tokens(1) / cost_ms(1);
        for (int t = 2; t <= cap; ++t) {
            // A few unlucky probes must not permanently price out a profitable
            // size. The bounded optimism decays with this request's trials;
            // it affects exploration, never the reported empirical expectation.
            const double bonus = request_probes_ ? .35 * (t - 1) / std::sqrt(std::max(1, request_trials_[t])) : 0;
            const double value = std::min<double>(t, expected_tokens(t) + bonus) / cost_ms(t);
            if (value > score * 1.03) { best = t; score = value; }
        }
        return best;
    }

    void observe(int t, int accepted, double verify_ms, double draft_ms) {
        if (t < 1 || t > cap_ || accepted < 0 || accepted >= t ||
            !std::isfinite(verify_ms) || !std::isfinite(draft_ms) ||
            !std::isfinite(verify_ms + draft_ms) || verify_ms <= 0 || draft_ms < 0)
            throw std::invalid_argument("invalid MTP budget observation");
        auto& s = sizes_[t];
        request_seen_[t] = true;
        ++request_trials_[t];
        if (s.observations++ > 0) {
            s.cost[s.next] = verify_ms + draft_ms;
            s.next = (s.next + 1) % int(s.cost.size());
            s.samples = std::min(s.samples + 1, int(s.cost.size()));
        }
        s.last = ++rounds_;
        for (int j = 1; j < t; ++j) {
            prefix_ok_[j] = .95 * prefix_ok_[j] + (accepted >= j ? 1.0 : 0.0);
            prefix_n_[j] = .95 * prefix_n_[j] + 1.0;
        }
    }

    double cost_ms(int t) const {
        auto sorted = sizes_.at(t).cost;
        const int n = sizes_.at(t).samples;
        if (!n) return 0;
        std::sort(sorted.begin(), sorted.begin() + n);
        return n % 2 ? sorted[n / 2] : .5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    }

    double expected_tokens(int t) const {
        double tokens = 1, previous = 1;
        for (int j = 1; j < t; ++j) {
            const double rate = prefix_n_[j] > 0 ? prefix_ok_[j] / prefix_n_[j] : 0;
            previous = std::min(previous, rate);
            tokens += previous;
        }
        return tokens;
    }

    std::string report() const {
        std::ostringstream out;
        out << "mtp budget: " << rounds_ << " observations;";
        for (int t = 1; t <= cap_; ++t)
            out << " T" << t << '=' << sizes_[t].observations << "@" << cost_ms(t)
                << "ms/" << expected_tokens(t) << "tok";
        return out.str();
    }

private:
    struct Size {
        std::array<double, 9> cost{};
        int samples = 0, next = 0;
        long long observations = 0, last = 0;
    };
    int cap_;
    long long rounds_ = 0;
    std::array<Size, kMaxT + 1> sizes_{};
    std::array<double, kMaxT + 1> prefix_ok_{}, prefix_n_{};
    std::array<bool, kMaxT + 1> request_seen_{};
    std::array<int, kMaxT + 1> request_trials_{};
    bool request_probes_ = false;
};

} // namespace strata::spec
