"""CPU-only regression suite for DPL Screening Skill; entry point is scripts/dpl.py."""
import json
import unittest
from pathlib import Path
import workflow_cases as cases

class WorkflowRegressionTests(unittest.TestCase):
    def test_v4_regressions(self):
        for ident,related,title,fn in cases.ENTRIES:
            with self.subTest(case=ident):
                cases.trial(ident,related,title,fn)
                result=cases.RESULTS[-1]
                self.assertTrue(result['executed'],result.get('harness_error'))
                self.assertTrue(result.get('detail',{}).get('passed'),result.get('detail'))
        cases.js(cases.OUT/'results.json',dict(target=str(cases.TARGET),target_sha256=cases.SOURCE_HASH,
            cases=cases.RESULTS,executed=sum(x['executed'] for x in cases.RESULTS),
            harness_errors=sum(not x['executed'] for x in cases.RESULTS),
            checks_meeting_full_expectation=sum(x.get('detail',{}).get('passed') is True for x in cases.RESULTS),
            real_gpu_jobs=0))

if __name__=='__main__':unittest.main()
