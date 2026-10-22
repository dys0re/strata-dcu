#include "strata/spec/mtp_budget.hpp"
#include <cstdio>
#include <limits>
#include <stdexcept>

using strata::spec::MtpBudget;
static void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
int main() {
    try {
        MtpBudget fast(5), rejected(5), interior(5);
        for (int t = 1; t <= 5; ++t) for (int r = 0; r < 12; ++r) {
            fast.observe(t, t - 1, 9 + t, .2 * (t - 1));
            rejected.observe(t, 0, 9 + t, 2 * (t - 1));
            interior.observe(t, t - 1, t <= 2 ? 10 + t : 100 * t, 0);
        }
        check(fast.choose(5) == 5, "cheap, accepted drafts should use the full window");
        check(rejected.choose(5) == 1, "rejected drafts must fall back to one real token");
        check(interior.choose(5) == 2, "expensive longer windows should select an interior optimum");
        MtpBudget expensive_draft(3);
        for (int t = 1; t <= 3; ++t) for (int r = 0; r < 8; ++r)
            expensive_draft.observe(t, t - 1, 10, 100 * (t - 1));
        check(expensive_draft.choose(3) == 1, "include draft generation, not only verifier time");
        check(fast.choose(2) == 2 && fast.choose(1) == 1, "respect output/context tail caps");
        MtpBudget cold(4);
        int seen[5] = {};
        for (int i = 0; i < 16; ++i) {
            int t = cold.choose(4); ++seen[t];
            cold.observe(t, t - 1, i == 0 ? 1e6 : 10 + t, 0);
        }
        for (int t = 1; t <= 4; ++t) check(seen[t] == 4, "every size needs measured warmup");
        check(cold.cost_ms(1) == 11, "discard first capture cost");
        bool probed = false;
        for (int i = 0; i < 160; ++i) {
            const int t = rejected.choose(5);
            probed |= t > 1;
            rejected.observe(t, 0, 9 + t, 2 * (t - 1));
        }
        check(probed, "remeasure stale sizes after workload changes");
        MtpBudget bad_request = rejected;
        bad_request.begin_request();
        for (int i = 0; i < 10; ++i) {
            const int t = bad_request.choose(5);
            bad_request.observe(t, 0, 9 + t, 2 * (t - 1));
        }
        check(bad_request.choose(5) == 1, "exploration also falls back after a new rejected task");
        rejected.begin_request();
        int fresh[6] = {};
        for (int i = 0; i < 10; ++i) {
            const int t = rejected.choose(5);
            ++fresh[t];
            rejected.observe(t, t - 1, 9 + t, 0);
        }
        for (int t = 1; t <= 5; ++t)
            check(fresh[t] == 2, "new request refreshes each size from the retained cost prior");
        check(rejected.expected_tokens(2) > 1.5, "new request acceptance overrides the previous rejected task");
        bool threw = false;
        try { cold.observe(2, 2, 10, 0); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "reject impossible acceptance counts");
        threw = false;
        try { cold.observe(1, 0, std::numeric_limits<double>::quiet_NaN(), 0); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "reject nonfinite runtime samples");
        std::puts("mtp_budget_test PASS: fallback, optimum, warmup, tail, drift, invalid input");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
