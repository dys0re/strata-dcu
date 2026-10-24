"""Generate an isolated unsigned-Q4 dot experiment; never replace engine objects."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
source=Path('/root/strata-dcu/Strata/src/kernels/cpu/q4_avx512.cpp')
s=source.read_text()
s=s.replace('q4_512_gu_rows(', 'angle_gu_rows(').replace('q4_512_rows(', 'angle_rows(').replace('q4_512_fits(', 'angle_fits(')
s=s.replace('__mmask32 ymin[kSlice][kMaxBlocks];','__m256i correction[kSlice][kMaxBlocks];')
start=s.index('                ymin[t][i] =')
end=s.index('\n            }',start)
s=s[:start]+'''                correction[t][i] = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                    _mm256_set1_epi8(8), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t][i].qs)));'''+s[end:]
start=s.index('inline __m256 product(')
end=s.index('\ntemplate<int NT',start)
s=s[:start]+'''inline __m256 product(__m256i q, __m256i, __m256i y, __m256i correction) {
    // (unsigned_nibble - 8) * signed_Q8. Input-only bias is shared by all rows.
    const __m256i p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q, y);
    return _mm256_cvtepi32_ps(_mm256_sub_epi32(p, correction));
}
'''+s[end:]
s=s.replace('_mm256_sub_epi8(unpack(g[i].qs), eight)','unpack(g[i].qs)')
s=s.replace('_mm256_sub_epi8(unpack(u[i].qs), eight)','unpack(u[i].qs)')
s=s.replace('a.ymin[t][i]','a.correction[t][i]')
(R/'q4_unsigned_probe.cpp').write_text(s)
(R/'probe-source.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(s.encode()).hexdigest()),indent=2)+'\n')
