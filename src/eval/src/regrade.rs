//! Regrading a captured evaluation trace without a model or device
//! (`regrade.cpp`, `trace.hpp`).

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvalCaseTrace {
    pub case_id: String,
    pub prompt: String,
    pub expected_answer: String,
    pub extracted_answer: String,
    pub generated_tokens: Vec<u32>,
    pub passed: bool,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvalTraceReport {
    pub model_id: String,
    pub route_identity: String,
    pub suite_hash: String,
    pub cases: Vec<EvalCaseTrace>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RegradeSummary {
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub pass_rate: f64,
}

/// `RegradeTrace`: a case passes when it passed, or when its extracted
/// answer equals a non-empty expected one.
pub fn regrade_trace(report: &EvalTraceReport) -> RegradeSummary {
    let mut summary = RegradeSummary {
        total_cases: report.cases.len(),
        ..RegradeSummary::default()
    };
    for case in &report.cases {
        if (!case.expected_answer.is_empty() && case.extracted_answer == case.expected_answer)
            || case.passed
        {
            summary.passed_cases += 1;
        } else {
            summary.failed_cases += 1;
        }
    }
    if summary.total_cases > 0 {
        summary.pass_rate = summary.passed_cases as f64 / summary.total_cases as f64;
    }
    summary
}
