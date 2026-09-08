"""Disposable local MFA site: never uses a real account or browser interception."""

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
