"""Trusted role templates and per-node provider-independent agent instances."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from atlas.agents import Agent
from atlas.plans.ir import AgentProfile
from atlas.plans.planner import PlannedNode, PlannerRole
from atlas.runtime.backend import AgentRegistration, ToolRegistration

logger = logging.getLogger("atlas.agent_factory")


class UnknownAgentRole(ValueError):
    """A plan requested a role that the trusted application did not register."""


@dataclass(frozen=True)
class AgentTemplate:
    """Operator-owned profile and constructor for one approved planner role."""

    role_id: str
    description: str
    profile: AgentProfile
    adapter_factory: Callable[[], Agent]


class AgentTemplateRegistry:
    """Immutable lookup of planner-visible roles and their trusted capabilities."""

    def __init__(
        self,
        templates: Mapping[str, AgentTemplate],
        *,
        tools: Mapping[str, ToolRegistration],
    ) -> None:
        resolved = dict(templates)
        for role_id, template in resolved.items():
            if role_id != template.role_id or template.profile.agent_id != role_id:
                raise ValueError("agent template role and profile IDs must match the registry key")
            unknown_tools = sorted(set(template.profile.allowed_tools) - set(tools))
            if unknown_tools:
                raise ValueError(
                    f"agent template '{role_id}' references unregistered tools: "
                    f"{', '.join(unknown_tools)}"
                )
        self._templates = resolved

    @property
    def planner_roles(self) -> tuple[PlannerRole, ...]:
        return tuple(
            PlannerRole(role_id=role.role_id, description=role.description)
            for role in sorted(self._templates.values(), key=lambda item: item.role_id)
        )

    def resolve(self, role_id: str) -> AgentTemplate:
        try:
            return self._templates[role_id]
        except KeyError:
            raise UnknownAgentRole(f"unknown trusted agent role '{role_id}'") from None


class AgentFactory:
    """Resolve trusted role IDs into per-node profiles and isolated adapter instances."""

    def __init__(self, templates: AgentTemplateRegistry) -> None:
        self.templates = templates

    def profiles_for(self, nodes: Iterable[PlannedNode]) -> dict[str, AgentProfile]:
        profiles: dict[str, AgentProfile] = {}
        for node in nodes:
            template = self.templates.resolve(node.role_id)
            instance_id = self._instance_id(node.role_id, node.node_id)
            profiles[node.node_id] = template.profile.model_copy(update={"agent_id": instance_id})
        return profiles

    async def instantiate_many(self, nodes: Iterable[PlannedNode]) -> dict[str, AgentRegistration]:
        registrations: dict[str, AgentRegistration] = {}
        adapters: set[int] = set()
        try:
            for node in nodes:
                template = self.templates.resolve(node.role_id)
                profile = template.profile.model_copy(
                    update={"agent_id": self._instance_id(node.role_id, node.node_id)}
                )
                adapter = template.adapter_factory()
                if id(adapter) in adapters:
                    raise ValueError("agent adapter factory returned a shared mutable instance")
                adapters.add(id(adapter))
                registrations[profile.agent_id] = AgentRegistration(profile, adapter)
        except BaseException:
            await self.aclose(registrations.values())
            raise
        return registrations

    async def aclose(self, registrations: Iterable[AgentRegistration]) -> None:
        seen: set[int] = set()
        for registration in registrations:
            adapter = registration.adapter
            if id(adapter) in seen:
                continue
            seen.add(id(adapter))
            closer = getattr(adapter, "aclose", None)
            if not callable(closer):
                closer = getattr(adapter, "close", None)
            if not callable(closer):
                continue
            try:
                result = closer()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                logger.error("agent_adapter_close_failed", extra={"error_type": type(exc).__name__})

    @staticmethod
    def _instance_id(role_id: str, node_id: str) -> str:
        return f"{role_id}__{node_id}"
