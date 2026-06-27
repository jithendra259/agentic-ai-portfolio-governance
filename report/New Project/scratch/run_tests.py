import unittest

suite = unittest.defaultTestLoader.discover('tests')
res = unittest.TestResult()
suite.run(res)

with open('test_summary.txt', 'w', encoding='utf-8') as fh:
    fh.write(f"Ran {res.testsRun} tests. Errors: {len(res.errors)}, Failures: {len(res.failures)}\n")
    for f in res.errors + res.failures:
        fh.write(f[0].id() + "\n")
        fh.write("\n".join(f[1].splitlines()[:10]) + "\n")
        fh.write("-" * 60 + "\n")
