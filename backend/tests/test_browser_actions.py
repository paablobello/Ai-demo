"""
Tests for BrowserActions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from browser.actions import BrowserActions


class MockProductConfig:
    """Mock ProductConfig for testing."""

    def __init__(self):
        self.clickable_elements = {
            "New Project": "button[data-testid='new-project']",
            "Submit": "button[type='submit']",
        }
        self.navigation_map = {
            "projects": "a[href='/projects']",
            "team": "a[href='/team']",
            "analytics": "a[href='/analytics']",
        }
        self.input_fields = {
            "project name": "input[name='project-name']",
            "email": "input[type='email']",
        }


class TestBrowserActions:
    """Tests for BrowserActions selector resolution."""

    @pytest.fixture
    def mock_controller(self):
        controller = MagicMock()
        controller.click = AsyncMock(return_value={"success": True})
        controller.type_text = AsyncMock(return_value={"success": True})
        controller.scroll = AsyncMock(return_value={"success": True})
        controller.highlight = AsyncMock(return_value={"success": True})
        controller.page = MagicMock()
        controller.page.wait_for_load_state = AsyncMock()
        return controller

    @pytest.fixture
    def actions(self, mock_controller):
        config = MockProductConfig()
        return BrowserActions(mock_controller, config)

    def test_resolve_selector_exact_match(self, actions):
        selector = actions._resolve_selector(
            "projects",
            actions.config.navigation_map
        )
        assert selector == "a[href='/projects']"

    def test_resolve_selector_case_insensitive(self, actions):
        selector = actions._resolve_selector(
            "PROJECTS",
            actions.config.navigation_map
        )
        assert selector == "a[href='/projects']"

    def test_resolve_selector_partial_match(self, actions):
        selector = actions._resolve_selector(
            "proj",
            actions.config.navigation_map
        )
        # Should match projects via partial
        assert "projects" in selector or selector == "a[href='/projects']"

    def test_resolve_selector_fallback(self, actions):
        selector = actions._resolve_selector(
            "nonexistent",
            actions.config.navigation_map
        )
        assert selector == "text=nonexistent"

    def test_generate_alternative_selectors_click(self, actions):
        selectors = actions._generate_alternative_selectors("New Project", selector_type="click")

        # Should include configured selector first
        assert "button[data-testid='new-project']" in selectors

        # Should include text-based selectors
        assert "text=New Project" in selectors

        # Should include role-based selectors
        assert any("role=button" in s for s in selectors)

    def test_generate_alternative_selectors_input(self, actions):
        selectors = actions._generate_alternative_selectors("email", selector_type="input")

        # Should include configured selector
        assert "input[type='email']" in selectors

        # Should include role-based input selectors
        assert any("role=textbox" in s for s in selectors)

        # Should include placeholder-based selectors
        assert any("placeholder" in s for s in selectors)

    def test_generate_alternative_selectors_no_duplicates(self, actions):
        selectors = actions._generate_alternative_selectors("Submit")

        # Should have no duplicates
        assert len(selectors) == len(set(selectors))

    @pytest.mark.asyncio
    async def test_navigate_to(self, actions, mock_controller):
        result = await actions.navigate_to("projects")

        assert result["success"] is True
        assert result["action"] == "navigate_to"
        assert result["destination"] == "projects"
        mock_controller.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_element_success(self, actions, mock_controller):
        result = await actions.click_element("New Project")

        assert result["success"] is True
        assert result["action"] == "click_element"
        assert result["element"] == "New Project"

    @pytest.mark.asyncio
    async def test_click_element_with_retry(self, actions, mock_controller):
        # First call fails, second succeeds
        mock_controller.click = AsyncMock(side_effect=[
            {"success": False, "error": "Not found"},
            {"success": True},
        ])

        result = await actions.click_element("New Project")

        assert result["success"] is True
        # Should have tried multiple selectors
        assert mock_controller.click.call_count >= 1

    @pytest.mark.asyncio
    async def test_type_text(self, actions, mock_controller):
        result = await actions.type_text("project name", "My Test Project")

        assert result["success"] is True
        assert result["action"] == "type_text"
        assert result["field"] == "project name"

    @pytest.mark.asyncio
    async def test_scroll_page(self, actions, mock_controller):
        result = await actions.scroll_page(direction="down", amount="medium")

        assert result["success"] is True
        assert result["action"] == "scroll_page"
        mock_controller.scroll.assert_called_once_with("down", "medium")

    @pytest.mark.asyncio
    async def test_highlight_element(self, actions, mock_controller):
        result = await actions.highlight_element("New Project")

        assert result["success"] is True
        assert result["action"] == "highlight_element"

    @pytest.mark.asyncio
    async def test_execute_action(self, actions, mock_controller):
        result = await actions.execute_action(
            "navigate_to",
            {"destination": "team"}
        )

        assert result["success"] is True
        assert result["destination"] == "team"

    @pytest.mark.asyncio
    async def test_execute_action_unknown(self, actions):
        result = await actions.execute_action(
            "unknown_action",
            {}
        )

        assert result["success"] is False
        assert "Unknown action" in result["error"]


class TestRetryMechanism:
    """Tests for retry mechanism with fallback selectors."""

    @pytest.fixture
    def mock_controller(self):
        controller = MagicMock()
        controller.page = MagicMock()
        controller.page.wait_for_load_state = AsyncMock()
        return controller

    @pytest.fixture
    def actions(self, mock_controller):
        config = MockProductConfig()
        return BrowserActions(mock_controller, config)

    @pytest.mark.asyncio
    async def test_retry_exhausts_all_selectors(self, actions, mock_controller):
        # All attempts fail
        mock_controller.click = AsyncMock(
            return_value={"success": False, "error": "Not found"}
        )

        result = await actions.click_element("NonexistentElement")

        assert result["success"] is False
        assert "tried_selectors" in result
        assert len(result["tried_selectors"]) == 3  # MAX_RETRIES

    @pytest.mark.asyncio
    async def test_retry_stops_on_success(self, actions, mock_controller):
        # Third attempt succeeds
        mock_controller.click = AsyncMock(side_effect=[
            {"success": False, "error": "Not found"},
            {"success": False, "error": "Not found"},
            {"success": True},
        ])

        result = await actions.click_element("Test")

        assert result["success"] is True
        assert mock_controller.click.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
