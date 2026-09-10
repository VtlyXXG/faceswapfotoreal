# Proposal frameworks — Upwork / Fiverr

Not ready-to-send text. Replace every `[bracket]` with details from the
actual job post before sending — a proposal that reads generic gets ignored
on these platforms even faster than on RU boards.

## Fiverr gig description (face-swap image service)

> I do face replacement on photos and covers with a production-style
> pipeline: face/landmark detection → head geometry alignment to the
> target → neural repaint (FLUX.2) → seam/background cleanup (LaMa) so the
> swap doesn't look pasted on. Good fit for personalized gifts, book
> covers, custom portraits. Send your face photo and the target image —
> I'll share a test render before you pay for the full order.

Do not add years of experience, client counts, or reviews that don't exist
yet — the paragraph above only claims what the pipeline actually does.

## Upwork proposal — small fixed-price image job

> [1 sentence showing you read the brief: what they want swapped, what
> matters most — likeness, angle, skin tone match]. I'll handle this with
> a pipeline that aligns head geometry to the target and cleans up the
> background after the swap, so there's no visible seam. Turnaround:
> [1–2 days]. I can send a test frame before final delivery.
>
> Quick question: [1 relevant question — final resolution needed / how
> many source photos / print or screen use].

## Upwork proposal — integration / dev job (face-swap feature for their app)

> [1 sentence on their actual need — API endpoint, batch processing,
> a specific model/tooling they mentioned]. I've built this exact kind of
> pipeline before: async task queue so the API never blocks on the GPU
> call, a storage layer behind a swappable driver contract, request
> tracing end to end. Happy to walk through the architecture on a call.
>
> To scope this precisely: [question about their current stack — existing
> API, expected volume, hosting/GPU access] and [question about deadline].

Rate guidance: quote from `../pricing.md`. For dev/integration work, default
to hourly ($25–40/h) unless the brief is precise enough to fix a price
confidently — an underscoped fixed price is a way to lose money on scope
creep, not a way to look competitive.

## When the client's budget is below range

Don't auto-discount. If the job is still worth taking (first review on a
new account, simple scope, potential repeat client), offer a smaller scope
instead of a flat discount:

> At that budget I can cover [1 image / no rush / no extra revisions] —
> the full scope with [extra] starts at [lower bound from pricing.md].
