FROM agenthack/base:1.1.1

ARG TEAM_ID=JTI
ARG AGENT_VERSION=0.1.0

ENV TEAM_ID="${TEAM_ID}"
ENV RESEARCH_WORKSPACE_TEAM="${TEAM_ID}"

WORKDIR /app

USER root
RUN rm -rf /app/src && mkdir -p /output && chown agent:agent /output && chmod 0755 /output
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=agent:agent agent.py /app/agent.py
COPY --chown=agent:agent src /app/src

LABEL org.agenthack.team_id="${TEAM_ID}"
LABEL org.agenthack.version="${AGENT_VERSION}"

USER agent
EXPOSE 8080
CMD ["python", "agent.py"]
