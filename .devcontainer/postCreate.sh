#!/bin/sh

# Fix permissions for the Go cache
sudo chown -R $(id -u):$(id -g) /var/cache/go

mkdir -p ~/.oh-my-zsh/custom/completions

# uv autocomplete
uv generate-shell-completion zsh >> ~/.oh-my-zsh/custom/completions/_uv

# uvx autocomplete
uvx --generate-shell-completion zsh >> ~/.oh-my-zsh/custom/completions/_uvx
