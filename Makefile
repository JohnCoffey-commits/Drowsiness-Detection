.PHONY: stage17-ui deployment-preflight

stage17-ui:
	./scripts/start_stage17_ui.sh

deployment-preflight:
	./scripts/deployment_preflight.sh
