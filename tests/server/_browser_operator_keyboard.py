"""Keyboard-only interaction with the rendered operator UI, without focus injection."""

from playwright.async_api import expect


class KeyboardOperator:
    def __init__(self, page):
        self.page = page
        self.activations = 0

    async def reach(self, target):
        # Native agent input can change the foreground tab in the shared test
        # browser. Restore the operator window, not an element's DOM focus.
        await self.page.bring_to_front()
        await expect(target).to_be_visible()
        await expect(target).to_be_enabled()
        for _ in range(100):
            if await target.evaluate("node => node === document.activeElement"):
                await expect(target).to_be_focused()
                return
            await self.page.keyboard.press("Tab")
        raise AssertionError("Operator control cannot be reached by keyboard.")

    async def activate(self, target):
        await self.reach(target)
        await self.page.keyboard.press("Enter")
        self.activations += 1

    async def deny_checkpoint(self, target):
        await self.reach(target)
        # Native select type-ahead also works with macOS headless Chrome,
        # whose OS popup does not consume synthetic arrow-key navigation.
        await self.page.keyboard.press("d")
        await self.page.keyboard.press("Tab")
        await expect(target).to_have_value("deny")

    async def type_private(self, target, text):
        await self.reach(target)
        await self.page.keyboard.type(text)
        await expect(target).to_have_value(text)
