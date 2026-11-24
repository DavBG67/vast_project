from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/seedvr2-ghcr-publish.yml")


def test_seedvr2_ghcr_publish_workflow_is_manual_only() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in content
    assert "\n  push:" not in content
    assert "\n  pull_request:" not in content


def test_seedvr2_ghcr_publish_workflow_uses_expected_image_and_tag_policy() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "DOCKERFILE_PATH: prototype/seedvr2/Dockerfile" in content
    assert "IMAGE_REPO: ghcr.io/davbg67/vast-video-upscaler-seedvr2" in content
    assert 'IMAGE_TAG="exp-6a2-${GITHUB_SHA}"' in content
    assert 'if [[ "${IMAGE_TAG}" == "latest" ]]; then' in content
    assert "ghcr.io/davbg67/vast-video-upscaler-worker:*" in content


def test_seedvr2_ghcr_publish_workflow_permissions_and_auth() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "contents: read" in content
    assert "packages: write" in content
    assert "uses: docker/login-action@v3" in content
    assert "password: ${{ secrets.GITHUB_TOKEN }}" in content


def test_seedvr2_ghcr_publish_workflow_verifies_build_and_published_image() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "Build SeedVR2 image before publish" in content
    assert 'docker build --file "$DOCKERFILE_PATH" --tag "${{ steps.prepare.outputs.image_ref }}" .' in content
    assert 'docker image inspect "${{ steps.prepare.outputs.image_ref }}" >/dev/null' in content
    assert "Push experimental image" in content
    assert 'docker push "${{ steps.prepare.outputs.image_ref }}"' in content
    assert 'docker run --rm "${{ steps.prepare.outputs.image_ref }}" --help | tee /tmp/seedvr2-runner-help.txt' in content
    assert "grep -q -- '--input-is-excerpt' /tmp/seedvr2-runner-help.txt" in content
    assert "Verify published image exists" in content
    assert 'docker buildx imagetools inspect "${{ steps.prepare.outputs.image_ref }}"' in content
    assert 'echo "- Published image: \\`${{ steps.prepare.outputs.image_ref }}\\`";' in content
