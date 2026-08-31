"""Vercel entrypoint package.

Present so ``api.index`` resolves as a regular package import rather than
relying on namespace-package discovery. It is not part of the distributable
package - ``[tool.setuptools.packages.find]`` restricts discovery to ``src``.
"""
