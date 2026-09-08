"""Disposable HTTPS password + TOTP fixture through Cayu's real operator panel."""

import asyncio
import hashlib
import hmac
import json
import secrets
import ssl
import struct
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse


def totp(key, counter):
    digest = hmac.digest(key, struct.pack(">Q", counter), "sha1")
    offset = digest[-1] & 15
    number = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return f"{number % 1000000:06d}"


class LocalMfa:
    origin: str
    tls: ssl.SSLContext

    def __init__(self):
        self.password = secrets.token_urlsafe(24)
        self.salt = secrets.token_bytes(16)
        self.password_hash = hashlib.pbkdf2_hmac(
            "sha256", self.password.encode(), self.salt, 100000
        )
        self.key = secrets.token_bytes(20)
        self.challenges = {}
        self.sessions = set()
        self.private_values = [self.password]
        self.password_checks = 0
        self.mfa_checks = 0
        self.rejected_mfa = 0
        self.completed = False
        self.api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        self.api.add_api_route("/fixture/login", self.login, methods=["GET"])
        self.api.add_api_route("/fixture/password", self.check_password, methods=["POST"])
        self.api.add_api_route("/fixture/mfa", self.mfa, methods=["GET"])
        self.api.add_api_route("/fixture/verify", self.verify, methods=["POST"])
        self.api.add_api_route("/fixture/member", self.member, methods=["GET"])

    async def forward(self, route):
        from urllib.parse import urlsplit

        import httpx

        headers = await route.request.all_headers()
        try:
            async with httpx.AsyncClient(verify=self.tls, trust_env=False) as client:
                response = await client.request(
                    route.request.method,
                    self.origin + urlsplit(route.request.url).path,
                    headers={
                        key: value
                        for key, value in headers.items()
                        if key in {"cookie", "content-type"}
                    },
                    content=route.request.post_data_buffer,
                )
        except Exception as error:
            print("Fixture transport failure: " + type(error).__name__, flush=True)
            raise
        outgoing = dict(response.headers)
        cookies = response.headers.get_list("set-cookie")
        if cookies:
            outgoing["set-cookie"] = "\n".join(cookies)
        await route.fulfill(status=response.status_code, headers=outgoing, body=response.content)

    async def check_rejections(self):
        import httpx

        async with httpx.AsyncClient(verify=self.tls, trust_env=False) as client:
            denied = await client.post(
                self.origin + "/fixture/password",
                json={"username": "disposable", "password": "deliberately-wrong"},
            )
            assert denied.status_code == 401
            denied = await client.post(
                self.origin + "/fixture/verify",
                json={"code": totp(self.key, int(time.time()) // 30)},
            )
            assert denied.status_code == 401
            denied = await client.get(self.origin + "/fixture/member")
            assert denied.status_code == 307
        assert self.password_checks == self.mfa_checks == 0

    def page(self, fields, endpoint):
        return HTMLResponse(
            """<!doctype html><html><body><h1>Disposable localhost MFA</h1>
        <form>"""
            + fields
            + """<button>Continue</button></form><p role="status"></p>
        <script>document.querySelector('form').onsubmit=async e=>{e.preventDefault();
        const form=e.target;const value=Object.fromEntries(new FormData(form));
        const response=await fetch("""
            + json.dumps(endpoint)
            + """,{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
        form.reset();if(response.ok)location.assign((await response.json()).next);
        else document.querySelector('[role=status]').textContent='Authentication rejected';};
        </script></body></html>""",
            headers={"Cache-Control": "no-store"},
        )

    async def login(self):
        return self.page(
            '<label>Username <input name="username" autofocus autocomplete="off"></label>'
            '<label>Password <input name="password" type="password" autocomplete="off"></label>',
            "/fixture/password",
        )

    async def check_password(self, request: Request):
        from fastapi.responses import JSONResponse

        data = await request.json()
        supplied = data.get("password", "")
        digest = hashlib.pbkdf2_hmac("sha256", str(supplied).encode(), self.salt, 100000)
        if data.get("username") != "disposable" or not hmac.compare_digest(
            digest, self.password_hash
        ):
            return JSONResponse({"error": "rejected"}, status_code=401)
        self.password_checks += 1
        challenge = secrets.token_urlsafe(32)
        self.challenges[challenge] = time.monotonic() + 120
        self.private_values.append(challenge)
        response = JSONResponse({"next": "/fixture/mfa"})
        response.set_cookie("mfa_pending", challenge, secure=True, httponly=True, samesite="strict")
        response.headers["Cache-Control"] = "no-store"
        return response

    async def mfa(self, request: Request):
        if self.challenges.get(request.cookies.get("mfa_pending"), 0) <= time.monotonic():
            return RedirectResponse("/fixture/login")
        return self.page(
            '<label>TOTP <input name="code" type="password" autofocus autocomplete="off"></label>',
            "/fixture/verify",
        )

    async def verify(self, request: Request):
        from fastapi.responses import JSONResponse

        challenge = request.cookies.get("mfa_pending")
        data = await request.json()
        code = str(data.get("code", ""))
        counter = int(time.time()) // 30
        valid = any(
            hmac.compare_digest(code, totp(self.key, counter + delta)) for delta in (-1, 0, 1)
        )
        if self.challenges.get(challenge, 0) <= time.monotonic() or not valid:
            self.rejected_mfa += 1
            return JSONResponse({"error": "rejected"}, status_code=401)
        del self.challenges[challenge]
        self.mfa_checks += 1
        token = secrets.token_urlsafe(32)
        self.sessions.add(token)
        self.private_values.extend([code, token])
        response = JSONResponse({"next": "/fixture/member"})
        response.delete_cookie("mfa_pending", secure=True, httponly=True, samesite="strict")
        response.set_cookie(
            "fixture_auth_cookie", token, secure=True, httponly=True, samesite="strict"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    async def member(self, request: Request):
        if request.cookies.get("fixture_auth_cookie") not in self.sessions:
            return RedirectResponse("/fixture/login")
        return HTMLResponse(
            "<h1>Authenticated localhost member</h1><p>Password and TOTP verified.</p>",
            headers={"Cache-Control": "no-store"},
        )

    def assert_safe(self, value):
        assert not any(secret in value for secret in self.private_values), (
            "Private fixture value escaped"
        )

    def assert_authenticated(self, value):
        self.assert_safe(value)
        assert self.password_checks == self.mfa_checks == 1
        assert "Authenticated localhost member" in value
        self.completed = True

    async def drive(self, panel, runtime, bound, *, native_page):
        from playwright.async_api import expect

        async def state():
            return (await runtime.coordinator._load(bound.record.identity))[1]

        async def until(predicate):
            async with asyncio.timeout(10):
                while not predicate(await state()):
                    await asyncio.sleep(0.05)

        await panel.get_by_role("button", name="Discover browsers").click()
        await panel.get_by_role("button", name="Browser 1 · agent_controlled").click()
        await panel.get_by_label("Profile checkpoint consent for the next takeover").select_option(
            "deny"
        )
        await panel.get_by_role("button", name="Request exclusive takeover").click()
        await until(lambda record: record.state == "operator_controlled")
        await panel.get_by_role("button", name="Refresh control state").click()
        await panel.get_by_role("button", name="Prepare sensitive entry").click()
        await until(lambda record: record.sensitive_entry and not record.sensitive_entry_pending)
        await panel.get_by_role("button", name="Refresh control state").click()

        async def send(value=None, key=None):
            await panel.get_by_role("button", name="Browser 1 · operator_controlled").click()
            current = await state()
            expected = current.settled_input_sequence + 1
            if key:
                await panel.get_by_role("button", name=key, exact=True).click()
            else:
                field = panel.get_by_label("Private value", exact=True)
                await field.fill(value)
                await panel.get_by_role("button", name="Send private text once").click()
                await expect(field).to_have_value("")
            await until(lambda record: record.settled_input_sequence == expected)

        await send("disposable")
        await send(key="Next field")
        await send(self.password)
        await send(key="Enter")
        async with asyncio.timeout(10):
            while not self.password_checks:
                await asyncio.sleep(0.05)
        # Wait for actual native navigation, not an arbitrary timing delay.
        await native_page.wait_for_url("https://mfa.example.test/fixture/mfa")
        rejected_before = self.rejected_mfa
        await send("not-a-totp")
        await send(key="Enter")
        async with asyncio.timeout(10):
            while self.rejected_mfa == rejected_before:
                await asyncio.sleep(0.05)
        assert self.mfa_checks == 0
        code = totp(self.key, int(time.time()) // 30)
        self.private_values.append(code)
        await send(code)
        await send(key="Enter")
        async with asyncio.timeout(10):
            while not self.mfa_checks:
                await asyncio.sleep(0.05)
        await native_page.wait_for_url("https://mfa.example.test/fixture/member")
        await panel.get_by_role("button", name="Return control to agent").click()
        await until(
            lambda record: record.state == "agent_controlled" and record.handback_audit is not None
        )
