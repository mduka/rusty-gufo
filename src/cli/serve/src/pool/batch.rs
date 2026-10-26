//! Batched decode across requests: the plan and the batched steps.

use super::*;

impl<R: TextModelRunner> TextRunnerPool<R> {
    /// The smallest batched plan covering `ready_requests`, else the widest,
    /// else serial.
    pub fn select_decode_plan(&self, ready_requests: usize) -> Result<TextExecutionPlan> {
        if ready_requests == 0 {
            return Err(Error::invalid(
                "decode plan requires at least one ready request",
            ));
        }
        if ready_requests == 1 {
            return Ok(TextExecutionPlan::SERIAL);
        }
        let batched = self
            .validated
            .plans
            .iter()
            .filter(|p| p.kind == TextExecutionPlanKind::Batched);
        let mut smallest_covering: Option<&TextExecutionPlan> = None;
        let mut widest: Option<&TextExecutionPlan> = None;
        for plan in batched {
            if widest.is_none_or(|w| plan.physical_width > w.physical_width) {
                widest = Some(plan);
            }
            if plan.physical_width >= ready_requests
                && smallest_covering.is_none_or(|s| plan.physical_width < s.physical_width)
            {
                smallest_covering = Some(plan);
            }
        }
        Ok(*smallest_covering
            .or(widest)
            .unwrap_or(&TextExecutionPlan::SERIAL))
    }

    pub(super) fn check_batched_plan(&self, count: usize, plan: &TextExecutionPlan) -> Result<()> {
        if plan.kind != TextExecutionPlanKind::Batched || count < 2 || count > plan.physical_width {
            return Err(Error::invalid("invalid batched text execution plan"));
        }
        if !self.validated.plans.contains(plan) {
            return Err(Error::invalid(
                "text runner does not support the requested batched plan",
            ));
        }
        Ok(())
    }

    /// Advances each request by its selected token in one runner call; the
    /// inner results are per request.
    pub fn advance_batch(
        &self,
        requests: &mut [&mut PoolRequest<R>],
        plan: &TextExecutionPlan,
    ) -> Result<Vec<Result<()>>> {
        self.check_batched_plan(requests.len(), plan)?;
        for request in requests.iter() {
            let inner = request
                .inner
                .as_ref()
                .ok_or_else(|| Error::logic("invalid request in batched text advance"))?;
            if !Arc::ptr_eq(&inner.runner, &self.validated.runner)
                || inner.pending_selection.is_none()
            {
                return Err(Error::logic("invalid request in batched text advance"));
            }
        }
        let mut inners = Vec::with_capacity(requests.len());
        for request in requests.iter_mut() {
            let inner = request.inner_mut()?;
            inner.capture_prompt_snapshot(true);
            inner.state_reusable = false;
            inners.push(inner);
        }
        let mut advances = Vec::with_capacity(inners.len());
        for inner in inners.iter_mut() {
            let token = inner
                .pending_selection
                .as_ref()
                .map(|s| s.token)
                .ok_or_else(|| Error::logic("invalid request in batched text advance"))?;
            advances.push(TextRunnerAdvance {
                state: inner.lease.state_mut()?,
                token,
            });
        }
        let results = self.validated.runner.advance_batch(&mut advances)?;
        drop(advances);
        if results.len() != inners.len() {
            return Err(Error::runtime(
                "text runner returned an invalid advance batch size",
            ));
        }
        for (inner, result) in inners.iter_mut().zip(&results) {
            if result.is_ok() {
                inner.pending_selection = None;
                inner.state_reusable = true;
            }
        }
        Ok(results)
    }

    /// One multi-token step per request in one runner call.
    pub fn decode_batch(
        &self,
        requests: &mut [&mut PoolRequest<R>],
        max_tokens: &[usize],
        plan: &TextExecutionPlan,
    ) -> Result<Vec<Result<TextDecodeStep>>> {
        if plan.kind != TextExecutionPlanKind::Batched
            || requests.len() < 2
            || requests.len() > plan.physical_width
            || max_tokens.len() != requests.len()
        {
            return Err(Error::invalid("invalid batched text decode plan"));
        }
        if !self.validated.plans.contains(plan) {
            return Err(Error::invalid(
                "text runner does not support the requested batched plan",
            ));
        }
        for (request, &budget) in requests.iter().zip(max_tokens) {
            let inner = request
                .inner
                .as_ref()
                .ok_or_else(|| Error::logic("invalid request in batched text decode"))?;
            if !Arc::ptr_eq(&inner.runner, &self.validated.runner)
                || !inner.decode_ready
                || inner.stopped
                || inner.pending_selection.is_some()
                || budget == 0
            {
                return Err(Error::logic("invalid request in batched text decode"));
            }
        }
        let mut inners = Vec::with_capacity(requests.len());
        for request in requests.iter_mut() {
            let inner = request.inner_mut()?;
            inner.capture_prompt_snapshot(true);
            inner.state_reusable = false;
            inners.push(inner);
        }
        let mut decodes = Vec::with_capacity(inners.len());
        for (inner, &budget) in inners.iter_mut().zip(max_tokens) {
            let RequestInner { lease, sampler, .. } = &mut **inner;
            decodes.push(TextRunnerDecode {
                state: lease.state_mut()?,
                max_tokens: budget,
                sampler,
            });
        }
        let steps = self.validated.runner.decode_batch(&mut decodes)?;
        drop(decodes);
        if steps.len() != inners.len() {
            return Err(Error::runtime(
                "text runner returned an invalid decode batch size",
            ));
        }
        for ((inner, step), &budget) in inners.iter_mut().zip(&steps).zip(max_tokens) {
            let Ok(step) = step else {
                continue;
            };
            let plan = step.execution_plan;
            if plan.physical_width == 0
                || plan.physical_width > max_tokens.len()
                || (plan.kind == TextExecutionPlanKind::Serial && plan.physical_width != 1)
            {
                return Err(Error::runtime(
                    "text runner returned an invalid execution width",
                ));
            }
            if step.selections.len() > budget || (step.selections.is_empty() && !step.stop) {
                return Err(Error::runtime(
                    "text runner returned an invalid batched decode step",
                ));
            }
            for selection in &step.selections {
                if selection.stop {
                    return Err(Error::runtime(
                        "text runner batched decode contains an embedded stop selection",
                    ));
                }
                inner.sampler.accept(selection.token);
                inner.generated.push(selection.token);
            }
            inner.stopped = step.stop;
            inner.state_reusable = true;
        }
        Ok(steps)
    }
}
