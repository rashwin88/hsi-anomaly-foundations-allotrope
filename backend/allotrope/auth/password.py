"""Password hashing using argon2id.

argon2-cffi's PasswordHasher encodes the algorithm, parameters, salt, and
hash into a single string (PHC format), so we store one column on User
(`password_hash`) and never have to manage salt/params separately.
"""

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

# Default parameters are deliberately strong — memory_cost=64MB, time_cost=3,
# parallelism=4. Override via PasswordHasher(...) only with a security
# review behind it.
_hasher = PasswordHasher()


def hash_password(plain: str) -> str:
    """Return the argon2id-encoded hash of `plain`.

    Output is a single string in PHC format that includes algorithm,
    version, params, salt, and hash — safe to store as `users.password_hash`.
    """
    return _hasher.hash(plain)


def verify_password(hashed: str, plain: str) -> bool:
    """Return True iff `plain` matches the stored `hashed`.

    Returns False on mismatch OR malformed hash. Never raises on the happy
    "wrong password" path; callers can branch on the boolean.
    """
    try:
        _hasher.verify(hashed, plain)
        return True
    except (VerifyMismatchError, InvalidHashError):
        return False
