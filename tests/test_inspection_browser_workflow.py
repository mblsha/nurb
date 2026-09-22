"""One real-browser pass through imported-reference inspection state."""

import asyncio
import contextlib
import threading

import pytest

from nurb import checks, compare, inspection, reference_import
from nurb.server import Server
from test_reference_import import assets, zipped


PART = """from nurb import *


@assembly
def thing():
    body = component(Box(20, 20, 2), "Body")
    lid = component(Pos(30, 0, 0) * Box(10, 10, 2), "Lid")
    return body, lid
"""


def live_server(server):
    """Run the socket, HTTP routes, and rebuild drain on one background loop."""
    from websockets.asyncio.server import serve

    up, done = threading.Event(), threading.Event()

    async def main():
        server.loop = asyncio.get_running_loop()
        server.queue = asyncio.Queue()
        drain = asyncio.create_task(server.drain())
        try:
            async with serve(
                server.ws,
                "127.0.0.1",
                server.port,
                process_request=server.http,
                origins=server.origins,
                open_timeout=None,
                max_size=70 * 1024 * 1024,
            ):
                up.set()
                await asyncio.to_thread(done.wait)
        finally:
            drain.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain

    thread = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
    thread.start()
    assert up.wait(timeout=10), "viewer server did not start"
    return done, thread


def test_textured_component_inspection_round_trips_through_the_real_viewer(tmp_path):
    pytest.importorskip("playwright", reason="nurb render is an optional extra")
    from playwright.sync_api import sync_playwright
    from nurb import render

    (tmp_path / "parts").mkdir()
    part = tmp_path / "parts" / "thing.py"
    part.write_text(PART)
    bundle = tmp_path / "scan.zip"
    bundle.write_bytes(zipped(assets()))

    server = Server(tmp_path)
    server.rebuild(part)
    server.port = render.free_port()
    done, thread = live_server(server)
    try:
        with sync_playwright() as playwright:
            browser = render._launch(playwright)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(f"http://127.0.0.1:{server.port}/?part=thing")
            page.wait_for_function("window.__nurb?.ready")

            page.locator("#ghostbtn").click()
            page.locator("#compareaddunits").select_option("cm")
            page.locator("#compareref").set_input_files(bundle)
            page.wait_for_function(
                """() => !document.querySelector('#importdetails').hidden
                    && document.querySelector('#importprovenance').textContent.includes('Texture preserved: texture.png')
                    && document.querySelector('#importprovenance').textContent.includes('2 component groups')""",
                timeout=30_000,
            )

            page.locator("#importdetails summary").click()
            page.locator("#importpreview").click()
            page.wait_for_function(
                "document.querySelectorAll('#importcomponentlist input').length === 2"
            )
            page.locator("#importcomponentlist input").nth(1).uncheck()
            assert "1 of 2" in page.locator("#importstatus").inner_text()
            page.locator("#importapply").click()
            page.wait_for_function(
                """() => document.querySelector('#importprovenance').textContent.includes('1 excluded')
                    && document.querySelector('#importstatus').textContent.includes('Comparison updated')""",
                timeout=30_000,
            )

            page.locator("#compareadvanced summary").first.click()
            page.locator("#comparecenter").click()
            page.wait_for_function(
                "document.querySelector('#comparestate').textContent.includes('Centered preview')"
            )
            centered = page.locator("#comparetransform input").evaluate_all(
                "inputs => inputs.map(input => input.value)"
            )
            assert any(abs(float(value)) > 0.01 for value in centered[:3])
            page.locator("#compareapply").click()
            page.wait_for_function(
                """() => !document.querySelector('#comparestate').textContent.includes('Saving alignment')
                    && document.querySelector('#compareapply').disabled""",
                timeout=30_000,
            )

            page.get_by_label("Isolate Body").click()
            assert page.get_by_label("Show Body").is_checked()
            assert not page.get_by_label("Show Lid").is_checked()

            page.locator("#regioneditor summary").first.click()
            page.locator("#regionname").fill("Body interface")
            page.locator("#regionkind").select_option("component")
            page.locator("#regioncomponent").select_option(label="Body")
            page.locator("#regionsave").click()
            page.wait_for_function(
                "document.querySelector('#regionstatus').textContent === 'Region saved.'",
                timeout=30_000,
            )
            page.get_by_role("button", name="Body interface", exact=True).click()
            page.locator("#comparemode").select_option("overlay")

            page.locator("#inspectionlabel").fill("Textured body fit")
            page.locator("#inspectionsave").click()
            page.wait_for_function(
                "document.querySelector('#inspectionstatus').textContent === 'Inspection setup saved in the project.'",
                timeout=30_000,
            )
            setup_id = page.locator("#inspectionchoice").input_value()
            assert setup_id

            page.locator("#componentsreset").click()
            page.locator("#comparemode").select_option("model")
            page.locator("#compareoriginal").click()
            page.get_by_role("button", name="Whole model").click()
            assert page.get_by_label("Show Lid").is_checked()

            page.locator("#inspectionrestore").click()
            page.wait_for_function(
                "document.querySelector('#inspectionstatus').textContent.startsWith('Restored Textured body fit.')",
                timeout=45_000,
            )
            assert page.locator("#comparemode").input_value() == "overlay"
            assert page.get_by_label("Show Body").is_checked()
            assert not page.get_by_label("Show Lid").is_checked()
            assert page.locator("#comparemetricstitle").inner_text() == "Region: Body interface"
            assert page.locator("#comparetransform input").evaluate_all(
                "inputs => inputs.map(input => input.value)"
            ) == centered
            browser.close()
    finally:
        done.set()
        thread.join(timeout=10)

    target = compare.setting(checks.settings(part))
    assert server.state["thing"]["target"]["import"]["texture"]["path"] == "texture.png"
    assert reference_import.manifest(tmp_path / target["file"])["excluded_components"] == [
        "component-2"
    ]
    saved = inspection.load(tmp_path, setup_id)
    assert saved["region"]["name"] == "Body interface"
    assert saved["region"]["component"] == "Body_1"
    assert saved["view"]["hidden_components"] == ["Lid_1"]
    assert saved["view"]["mode"] == "overlay"
    assert saved["view"]["alignment"] == target["transform"]


def test_fresh_model_without_reference_disables_evidence_actions_in_the_real_viewer(tmp_path):
    pytest.importorskip("playwright", reason="nurb render is an optional extra")
    from playwright.sync_api import sync_playwright
    from nurb import render

    (tmp_path / "parts").mkdir()
    part = tmp_path / "parts" / "thing.py"
    part.write_text(PART)
    server = Server(tmp_path)
    server.rebuild(part)
    server.port = render.free_port()
    done, thread = live_server(server)
    try:
        with sync_playwright() as playwright:
            browser = render._launch(playwright)
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            page.goto(f"http://127.0.0.1:{server.port}/?part=thing")
            page.wait_for_function("window.__nurb?.ready")
            page.locator("#ghostbtn").click()
            assert page.locator("#featureinspect").is_disabled()
            assert page.locator("#verifyrun").is_disabled()
            assert "Attach a reference" in page.locator("#evidenceguidance").inner_text()
            browser.close()
    finally:
        done.set()
        thread.join(timeout=10)
