const fs = require("fs");

module.exports = async ({ github, context }) => {
  const marker = "<!-- codex:rustdoc-missing-docs-debt -->";
  const rustdocStatus = process.env.RUSTDOC_STATUS;
  const summaryPath = process.env.SUMMARY_PATH;

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

  if (rustdocStatus === "clean") {
    if (existing) {
      await github.rest.issues.deleteComment({
        owner,
        repo,
        comment_id: existing.id,
      });
    }
    return;
  }

  let summary = "### Rustdoc Missing Docs Debt\n\n- Status: unavailable";
  if (summaryPath) {
    try {
      summary = fs.readFileSync(summaryPath, "utf8").trim();
    } catch {
      // Keep fallback summary.
    }
  }

  const body = [
    marker,
    summary,
    "",
    "- This is a non-blocking reminder posted from `docs-and-quality`.",
  ].join("\n");

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
