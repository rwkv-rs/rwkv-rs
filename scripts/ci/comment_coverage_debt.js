module.exports = async ({ github, context }) => {
  const marker = "<!-- codex:coverage-debt -->";
  const coverageStatus = process.env.COVERAGE_STATUS;
  const coveragePct = process.env.COVERAGE_PCT;
  const coverageTarget = process.env.COVERAGE_TARGET;

  const { owner, repo } = context.repo;
  const issue_number = context.issue.number;

  if (!issue_number) {
    return;
  }

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number,
    per_page: 100,
  });

  const existing = comments.find(
    (comment) => comment.user?.type === "Bot" && comment.body?.includes(marker),
  );

  if (coverageStatus === "target_met" || coverageStatus === "tests_failed") {
    if (existing) {
      await github.rest.issues.deleteComment({
        owner,
        repo,
        comment_id: existing.id,
      });
    }
    return;
  }

  const lines = [marker, "### Coverage Debt", ""];

  if (coverageStatus === "below_target") {
    lines.push(`- Workspace line coverage: ${coveragePct}%`);
    lines.push(`- Target: ${coverageTarget}%`);
    lines.push("- Status: non-blocking warning");
    lines.push(
      "- Tests passed, but coverage is below target and should be treated as debt.",
    );
  } else {
    lines.push("- Status: unavailable");
    lines.push("- Tests passed, but coverage reports could not be generated.");
    lines.push("- This is non-blocking debt and should be fixed.");
  }

  const body = lines.join("\n");

  if (existing) {
    await github.rest.issues.updateComment({
      owner,
      repo,
      comment_id: existing.id,
      body,
    });
  } else {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number,
      body,
    });
  }
};
