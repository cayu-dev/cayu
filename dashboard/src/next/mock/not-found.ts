/** What the fake API throws for an ID it doesn't know, like the server's 404. */
export class NotFound extends Error {
  constructor(what: string) {
    super(`${what} doesn't exist.`)
  }
}
