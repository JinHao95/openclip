import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  clampBoundsWithinRange,
  emptyProject,
  formatApiTimestamp,
  formatTimestamp,
  getDirtyState,
  getProjectIdFromPath,
  projectFromManifest,
  type ClipDraft,
  type EditorProject,
  type SubtitleSegmentDraft,
  serializeSubtitleSegments,
  subtitleSegmentsToText,
} from './editorState'
import { t, type Locale, type MessageKey } from './i18n'

const editorEndpoints = [
  'GET /api/projects/:project_id',
  'PATCH /api/projects/:project_id/clips/:clip_id/bounds',
  'PATCH /api/projects/:project_id/clips/:clip_id/subtitle',
  'PATCH /api/projects/:project_id/clips/:clip_id/translated-subtitle',
  'PATCH /api/projects/:project_id/clips/:clip_id/cover-title',
  'POST /api/projects/:project_id/clips/:clip_id/rerender/boundary',
  'POST /api/projects/:project_id/clips/:clip_id/rerender/subtitles',
  'POST /api/projects/:project_id/clips/:clip_id/rerender/cover',
  'POST /api/projects/:project_id/clips/:clip_id/resume',
  'GET /api/jobs/:job_id',
]

const emptyDirtyState = {
  hasChanges: false,
  boundsDirty: false,
  speedDirty: false,
  boundaryDirty: false,
  sourceSubtitlesDirty: false,
  translatedSubtitlesDirty: false,
  subtitlesDirty: false,
  coverTitleDirty: false,
  coverNeedsRefresh: false,
}

const speedOptions = [1, 1.25, 1.5, 2, 3]

const jobPollIntervalMs = 1000
const jobPollTimeoutMs = 5 * 60 * 1000
const reconciliationPollIntervalMs = 300
const reconciliationTimeoutMs = 3000
const loadProjectFailedError = 'openclip_load_project_failed'

interface LoadedProjectResult {
  project: EditorProject
  statusMessage: UiText
  logMessage: UiText
}

interface PreviewWindow {
  start: number
  end: number
  sourceVideoUrl?: string
}

type EditorTheme = 'dark' | 'light'
type MessageValues = Record<string, string | number>
type UiText = { key: MessageKey; values?: MessageValues } | { raw: string }

const themeStorageKey = 'openclip-editor-theme'
const languageStorageKey = 'openclip-editor-language'

function loadInitialTheme(): EditorTheme {
  try {
    const storedTheme = window.localStorage.getItem(themeStorageKey)
    return storedTheme === 'dark' ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

function loadInitialLocale(): Locale {
  try {
    const storedLocale = window.localStorage.getItem(languageStorageKey)
    return storedLocale === 'en' ? 'en' : 'zh'
  } catch {
    return 'zh'
  }
}

function withVersionToken(url?: string, version?: string): string | undefined {
  if (!url) {
    return undefined
  }
  if (!version) {
    return url
  }
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}v=${encodeURIComponent(version)}`
}

function message(key: MessageKey, values?: MessageValues): UiText {
  return { key, values }
}

function rawMessage(text: string): UiText {
  return { raw: text }
}

function formatSpeedLabel(speed: number): string {
  return `${speed}x`
}

function buildPreviewWindow(clip?: ClipDraft, projectSourceVideoUrl?: string): PreviewWindow | null {
  if (!clip) {
    return null
  }
  return {
    start: clip.start,
    end: clip.end,
    sourceVideoUrl: clip.sourceVideoUrl ?? projectSourceVideoUrl,
  }
}

function loadFailureMessage(error: unknown, projectId: string): UiText {
  if (error instanceof Error && error.message === loadProjectFailedError) {
    return message('unableLoadManifestProject', { projectId })
  }
  if (error instanceof Error) {
    return rawMessage(error.message)
  }
  return message('unableLoadProject')
}

function subtitleTextareaRows(text: string): number {
  const estimatedRows = text.split('\n').reduce((rows, line) => {
    return rows + Math.max(1, Math.ceil(Math.max(line.length, 1) / 56))
  }, 0)
  return Math.max(2, Math.min(3, estimatedRows))
}

function App() {
  const projectId = useMemo(() => getProjectIdFromPath(window.location.pathname), [])
  const [savedProject, setSavedProject] = useState<EditorProject>(() => emptyProject(projectId))
  const [draftProject, setDraftProject] = useState<EditorProject>(() => emptyProject(projectId))
  const [activeClipId, setActiveClipId] = useState('')
  const [activityLog, setActivityLog] = useState<UiText[]>([])
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<UiText | null>(null)
  const [statusMessage, setStatusMessage] = useState<UiText>(message('loadingEditorProject'))
  const [dragHandle, setDragHandle] = useState<'start' | 'end' | null>(null)
  const [theme, setTheme] = useState<EditorTheme>(loadInitialTheme)
  const [locale, setLocale] = useState<Locale>(loadInitialLocale)
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false)

  const timelineTrackRef = useRef<HTMLDivElement | null>(null)
  const previewVideoRef = useRef<HTMLVideoElement | null>(null)
  const subtitlePreviewVideoRef = useRef<HTMLVideoElement | null>(null)
  const activeClipIdRef = useRef('')

  const savedClipMap = useMemo(() => new Map(savedProject.clips.map((clip) => [clip.id, clip])), [savedProject.clips])
  const activeClip = draftProject.clips.find((clip) => clip.id === activeClipId) ?? draftProject.clips[0]
  const savedActiveClip = savedClipMap.get(activeClip?.id ?? '') ?? activeClip
  const subtitlePreviewUrl = withVersionToken(activeClip?.currentComposedClipUrl, activeClip?.updatedAt)
  const horizontalCoverPreviewUrl = withVersionToken(activeClip?.horizontalCoverUrl, activeClip?.updatedAt)
  const verticalCoverPreviewUrl = withVersionToken(activeClip?.verticalCoverUrl, activeClip?.updatedAt)
  const previewWindow = useMemo(
    () => buildPreviewWindow(activeClip, draftProject.sourceVideoUrl),
    [activeClip, draftProject.sourceVideoUrl],
  )
  const activeDirtyState = activeClip && savedActiveClip ? getDirtyState(savedActiveClip, activeClip) : emptyDirtyState
  const activePartStart = activeClip?.partAbsoluteStart ?? 0
  const activePartEnd = activeClip?.partAbsoluteEnd ?? draftProject.totalDuration
  const outputDuration = activeClip ? (activeClip.end - activeClip.start) / Math.max(activeClip.speed, 0.001) : 0
  const TIMELINE_PADDING_SECONDS = 120
  const timelineViewStart = Math.max(0, (activeClip?.start ?? 0) - TIMELINE_PADDING_SECONDS)
  const timelineViewEnd = Math.min(draftProject.totalDuration, (activeClip?.end ?? 0) + TIMELINE_PADDING_SECONDS)
  const timelineViewDuration = timelineViewEnd - timelineViewStart
  const dirtyClipCount = draftProject.clips.filter((clip) => {
    const savedClip = savedClipMap.get(clip.id)
    return savedClip ? getDirtyState(savedClip, clip).hasChanges || Boolean(clip.coverDirty) : false
  }).length

  const resolveText = useCallback((entry: UiText) => (
    'raw' in entry ? entry.raw : t(locale, entry.key, entry.values)
  ), [locale])

  const pushLog = useCallback((entry: UiText) => {
    setActivityLog((current) => [entry, ...current].slice(0, 8))
  }, [])

  const applyLoadedProject = useCallback((result: LoadedProjectResult) => {
    const targetClip = result.project.clips.find((clip) => clip.id === activeClipIdRef.current) ?? result.project.clips[0]
    setSavedProject(result.project)
    setDraftProject(result.project)
    setActiveClipId(targetClip?.id ?? '')
    setStatusMessage(result.statusMessage)
    setLoadError(null)
    pushLog(result.logMessage)
    setLoading(false)
  }, [pushLog])

  const loadProject = useCallback(async (): Promise<LoadedProjectResult> => {
    const response = await fetch(`/api/projects/${projectId}`)
    if (!response.ok) {
      throw new Error(loadProjectFailedError)
    }
    const manifest = await response.json()
    const project = projectFromManifest(manifest)
    return {
      project,
      statusMessage: message('loadedManifestProjectStatus', { projectName: project.projectName }),
      logMessage: message('loadedManifestProjectLog', { projectId }),
    }
  }, [projectId])

  useEffect(() => {
    let cancelled = false

    async function initializeProject() {
      try {
        const result = await loadProject()
        if (!cancelled) {
          applyLoadedProject(result)
        }
      } catch (error) {
        if (!cancelled) {
          const nextMessage = loadFailureMessage(error, projectId)
          setLoadError(nextMessage)
          setStatusMessage(nextMessage)
          setLoading(false)
          pushLog(nextMessage)
        }
      }
    }

    void initializeProject()

    return () => {
      cancelled = true
    }
  }, [applyLoadedProject, loadProject, projectId, pushLog])

  useEffect(() => {
    try {
      window.localStorage.setItem(themeStorageKey, theme)
    } catch {
      // Ignore persistence failures and keep the in-memory theme.
    }
    document.documentElement.style.colorScheme = theme
  }, [theme])

  useEffect(() => {
    try {
      window.localStorage.setItem(languageStorageKey, locale)
    } catch {
      // Ignore persistence failures and keep the in-memory locale.
    }
  }, [locale])

  useEffect(() => {
    activeClipIdRef.current = activeClipId
  }, [activeClipId])

  useEffect(() => {
    if (!dragHandle || !activeClip || !timelineTrackRef.current) {
      return undefined
    }

    function handlePointerMove(event: PointerEvent) {
      const rect = timelineTrackRef.current?.getBoundingClientRect()
      if (!rect || timelineViewDuration <= 0) {
        return
      }

      const ratio = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1)
      const nextSeconds = timelineViewStart + ratio * timelineViewDuration
      const clamped = dragHandle === 'start'
        ? clampBoundsWithinRange(nextSeconds, activeClip.end, draftProject.totalDuration, activePartStart, activePartEnd ?? draftProject.totalDuration)
        : clampBoundsWithinRange(activeClip.start, nextSeconds, draftProject.totalDuration, activePartStart, activePartEnd ?? draftProject.totalDuration)
      setDraftProject((current) => ({
        ...current,
        clips: current.clips.map((clip) => (
          clip.id === activeClip.id ? { ...clip, start: clamped.start, end: clamped.end, coverDirty: true } : clip
        )),
      }))
    }

    function handlePointerUp() {
      setDragHandle(null)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)
    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
    }
  }, [activeClip, activePartEnd, activePartStart, dragHandle, draftProject.totalDuration, timelineViewStart, timelineViewDuration])

  useEffect(() => {
    if (!previewVideoRef.current || !previewWindow) {
      return
    }
    const currentVideo = previewVideoRef.current
    const currentWindow = previewWindow

    function handleLoadedMetadata() {
      currentVideo.currentTime = currentWindow.start
      currentVideo.pause()
    }

    function handleTimeUpdate() {
      if (currentVideo.currentTime >= currentWindow.end) {
        currentVideo.pause()
      }
    }

    try {
      currentVideo.pause()
      currentVideo.currentTime = currentWindow.start
    } catch {
      // If metadata is not ready yet, loadedmetadata will apply the same previewWindow.
    }

    currentVideo.addEventListener('loadedmetadata', handleLoadedMetadata)
    currentVideo.addEventListener('timeupdate', handleTimeUpdate)
    return () => {
      currentVideo.removeEventListener('loadedmetadata', handleLoadedMetadata)
      currentVideo.removeEventListener('timeupdate', handleTimeUpdate)
    }
  }, [previewWindow])

  useEffect(() => {
    if (!subtitlePreviewVideoRef.current || !subtitlePreviewUrl) {
      return
    }

    const video = subtitlePreviewVideoRef.current
    const previewOffset = 0.05

    function showFirstFrame() {
      if (!Number.isFinite(video.duration) || video.duration <= 0) {
        return
      }
      const targetTime = Math.min(previewOffset, Math.max(video.duration - 0.001, 0))
      if (targetTime <= 0) {
        return
      }
      try {
        video.currentTime = targetTime
      } catch {
        // Some browsers may reject early seeks until the media is a little further along.
      }
    }

    function handleSeeked() {
      video.pause()
    }

    video.addEventListener('loadedmetadata', showFirstFrame)
    video.addEventListener('seeked', handleSeeked)

    if (video.readyState >= 1) {
      showFirstFrame()
    }

    return () => {
      video.removeEventListener('loadedmetadata', showFirstFrame)
      video.removeEventListener('seeked', handleSeeked)
    }
  }, [subtitlePreviewUrl])

  function getOperationLabel(action: 'boundary' | 'subtitles' | 'cover'): string {
    const operationKeys: Record<'boundary' | 'subtitles' | 'cover', MessageKey> = {
      boundary: 'operationBoundary',
      subtitles: 'operationSubtitle',
      cover: 'operationCover',
    }
    return t(locale, operationKeys[action])
  }

  function getDirtyStateLabel(state: typeof emptyDirtyState): string {
    if (!state.hasChanges && !state.coverNeedsRefresh) {
      return t(locale, 'noLocalChanges')
    }
    const tokens = []
    if (state.boundsDirty) tokens.push(t(locale, 'dirtyTokenBounds'))
    if (state.speedDirty) tokens.push(t(locale, 'dirtyTokenSpeed'))
    if (state.subtitlesDirty) tokens.push(t(locale, 'dirtyTokenSubtitles'))
    if (state.coverNeedsRefresh) tokens.push(t(locale, 'dirtyTokenCover'))
    return t(locale, 'dirtySummary', { items: tokens.join(' + ') })
  }

  function getRenderStatusLabel(status: ClipDraft['renderStatus']): string {
    const renderStatusKeys: Record<ClipDraft['renderStatus'], MessageKey> = {
      Ready: 'renderStatusReady',
      'Needs sync': 'renderStatusNeedsSync',
      Rendering: 'renderStatusRendering',
      Recoverable: 'renderStatusRecoverable',
      Error: 'renderStatusError',
    }
    return t(locale, renderStatusKeys[status])
  }

  async function assertOk(response: Response, action: string) {
    if (!response.ok) {
      throw new Error(`${action} failed with status ${response.status}`)
    }
  }

  function updateClip(id: string, updater: (clip: ClipDraft) => ClipDraft) {
    setDraftProject((current) => ({
      ...current,
      clips: current.clips.map((clip) => (clip.id === id ? updater(clip) : clip)),
    }))
  }

  function updateBoundsLocally(nextStart: number, nextEnd: number) {
    if (!activeClip) return
    const clamped = clampBoundsWithinRange(
      nextStart,
      nextEnd,
      draftProject.totalDuration,
      activePartStart,
      activePartEnd ?? draftProject.totalDuration,
    )
    updateClip(activeClip.id, (clip) => ({
      ...clip,
      start: clamped.start,
      end: clamped.end,
      coverDirty: true,
    }))
  }

  async function patchClip(clip: ClipDraft) {
    const savedClip = savedClipMap.get(clip.id)
    if (!savedClip) return

    if (savedClip.start !== clip.start || savedClip.end !== clip.end || savedClip.speed !== clip.speed) {
      const response = await fetch(`/api/projects/${projectId}/clips/${clip.id}/bounds`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_time: formatApiTimestamp(clip.start),
          end_time: formatApiTimestamp(clip.end),
          speed: clip.speed,
        }),
      })
      await assertOk(response, 'Saving clip bounds')
    }
    if (serializeSubtitleSegments(savedClip.subtitleSegments) !== serializeSubtitleSegments(clip.subtitleSegments)) {
      const response = await fetch(`/api/projects/${projectId}/clips/${clip.id}/subtitle`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subtitle_text: clip.subtitleText,
          subtitle_segments: clip.subtitleSegments.map((segment) => ({
            start_time: segment.startTime,
            end_time: segment.endTime,
            text: segment.text,
          })),
        }),
      })
      await assertOk(response, 'Saving subtitle override')
    }
    if (serializeSubtitleSegments(savedClip.translatedSubtitleSegments) !== serializeSubtitleSegments(clip.translatedSubtitleSegments)) {
      const response = await fetch(`/api/projects/${projectId}/clips/${clip.id}/translated-subtitle`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subtitle_text: clip.translatedSubtitleText,
          subtitle_segments: clip.translatedSubtitleSegments.map((segment) => ({
            start_time: segment.startTime,
            end_time: segment.endTime,
            text: segment.text,
          })),
        }),
      })
      await assertOk(response, 'Saving translated subtitle override')
    }
    if (savedClip.coverTitle !== clip.coverTitle) {
      const response = await fetch(`/api/projects/${projectId}/clips/${clip.id}/cover-title`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title_text: clip.coverTitle }),
      })
      await assertOk(response, 'Saving cover title')
    }
  }

  async function loadProjectUntilJobReconciled(jobId: string, clipId: string): Promise<LoadedProjectResult> {
    let latestResult: LoadedProjectResult | null = null
    const deadline = Date.now() + reconciliationTimeoutMs
    while (Date.now() < deadline) {
      latestResult = await loadProject()
      const latestClip = latestResult.project.clips.find((clip) => clip.id === clipId)
      if (!latestClip?.pendingJobId || latestClip.pendingJobId !== jobId) {
        return latestResult
      }
      await new Promise((resolve) => setTimeout(resolve, reconciliationPollIntervalMs))
    }
    return latestResult ?? loadProject()
  }

  function handleResetClip() {
    if (!activeClip) return
    const savedClip = savedClipMap.get(activeClip.id)
    if (!savedClip) return
    updateClip(activeClip.id, () => ({ ...savedClip }))
    pushLog(message('resetClipLog', { title: activeClip.title }))
  }

  async function pollJob(jobId: string, clipId: string) {
    const deadline = Date.now() + jobPollTimeoutMs
    while (Date.now() < deadline) {
      const response = await fetch(`/api/jobs/${jobId}`)
      if (response.ok) {
        const job = await response.json()
        const status = typeof job.status === 'string' ? job.status : job.status?.value
        if (status === 'completed') {
          pushLog(message('jobCompletedLog', { jobId }))
          applyLoadedProject(await loadProjectUntilJobReconciled(jobId, clipId))
          return
        }
        if (status === 'failed' || status === 'cancelled') {
          updateClip(clipId, (clip) => ({
            ...clip,
            renderStatus: 'Error',
            lastError: job.error ?? `Job ${status}.`,
            pendingJobId: undefined,
            pendingOperation: undefined,
          }))
          setStatusMessage(job.error ? rawMessage(job.error) : message('jobEndedStatus', { status }))
          pushLog(message('jobEndedLog', { jobId, status }))
          return
        }
      }
      await new Promise((resolve) => setTimeout(resolve, jobPollIntervalMs))
    }

    updateClip(clipId, (clip) => ({
      ...clip,
      renderStatus: 'Error',
      lastError: t(locale, 'timedOutWaitingRerenderStatus'),
      pendingJobId: undefined,
      pendingOperation: undefined,
    }))
    setStatusMessage(message('timedOutWaitingRerenderStatus'))
    pushLog(message('timedOutWaitingJobLog', { jobId }))
  }

  async function handleQueue(action: 'boundary' | 'subtitles' | 'cover') {
    if (!activeClip || activeClip.renderStatus === 'Rendering') return
    try {
      await patchClip(activeClip)
      const response = await fetch(`/api/projects/${projectId}/clips/${activeClip.id}/rerender/${action}`, { method: 'POST' })
      await assertOk(response, t(locale, 'queueRerender', { operation: getOperationLabel(action) }))
      const payload = await response.json()
      const queuedStatus = message('queuedRerenderStatus', { operation: getOperationLabel(action), title: activeClip.title })
      setStatusMessage(queuedStatus)
      pushLog(message('queuedRerenderLog', { operation: getOperationLabel(action), jobId: payload.job_id }))
      if (action === 'boundary') {
        applyLoadedProject(await loadProject())
        setStatusMessage(queuedStatus)
      } else {
        updateClip(activeClip.id, (clip) => ({
          ...clip,
          renderStatus: 'Rendering',
          pendingJobId: payload.job_id,
          pendingOperation: action,
          lastError: undefined,
        }))
      }
      void pollJob(payload.job_id, activeClip.id)
    } catch (error) {
      const nextMessage = error instanceof Error ? error.message : t(locale, 'unableQueueRerender', { operation: getOperationLabel(action) })
      updateClip(activeClip.id, (clip) => ({ ...clip, renderStatus: 'Error', lastError: nextMessage }))
      setStatusMessage(rawMessage(nextMessage))
      pushLog(rawMessage(nextMessage))
    }
  }

  async function handleResume() {
    if (!activeClip) return
    try {
      const response = await fetch(`/api/projects/${projectId}/clips/${activeClip.id}/resume`, { method: 'POST' })
      await assertOk(response, t(locale, 'resumingRerender'))
      const payload = await response.json()
      pushLog(message('resumedRerenderLog', { operation: getOperationLabel(payload.operation), jobId: payload.job_id }))
      updateClip(activeClip.id, (clip) => ({
        ...clip,
        renderStatus: 'Rendering',
        pendingJobId: payload.job_id,
        lastError: undefined,
      }))
      void pollJob(payload.job_id, activeClip.id)
    } catch (error) {
      const nextMessage = error instanceof Error ? error.message : t(locale, 'unableResumeRerender')
      setStatusMessage(rawMessage(nextMessage))
      pushLog(rawMessage(nextMessage))
    }
  }

  if (loadError && draftProject.clips.length === 0) {
    return (
      <div className="shell">
        <main className="workspace-grid">
          <section className="panel editor-pane" aria-label={t(locale, 'editorLoadFailure')}>
            <div className="panel__header panel__header--stacked">
              <div>
                <p className="eyebrow">{t(locale, 'openClipEditor')}</p>
                <h2>{t(locale, 'editorUnavailable')}</h2>
                <p className="muted">{resolveText(loadError)}</p>
              </div>
            </div>
            <footer className="editor-actions">
              <button type="button" className="action-button" onClick={() => {
                setLoading(true)
                setLoadError(null)
                void loadProject().then(applyLoadedProject).catch((error) => {
                  const nextMessage = loadFailureMessage(error, projectId)
                  setLoadError(nextMessage)
                  setStatusMessage(nextMessage)
                  setLoading(false)
                  pushLog(nextMessage)
                })
              }}>{t(locale, 'retryLoad')}</button>
            </footer>
          </section>
        </main>
      </div>
    )
  }

  if (!activeClip) {
    return <div className="shell"><main className="workspace-grid"><p>{t(locale, 'noClipsAvailable')}</p></main></div>
  }

  const timelineStart = timelineViewDuration > 0 ? ((activeClip.start - timelineViewStart) / timelineViewDuration) * 100 : 0
  const timelineWidth = timelineViewDuration > 0 ? ((activeClip.end - activeClip.start) / timelineViewDuration) * 100 : 0
  const previewSourceUrl = previewWindow?.sourceVideoUrl ?? activeClip.sourceVideoUrl ?? draftProject.sourceVideoUrl
  const diagnosticsStatusKind = loading ? 'loading' : loadError ? 'error' : 'connected'
  const diagnosticsStatus = loading ? t(locale, 'loadingStatus') : loadError ? t(locale, 'errorStatus') : t(locale, 'connectedStatus')
  const activePendingOperation = activeClip.pendingOperation as 'boundary' | 'subtitles' | 'cover' | undefined
  const shouldShowStatusBanner = loading
    || activeClip.renderStatus === 'Recoverable'
    || activeClip.renderStatus === 'Error'
    || activeDirtyState.hasChanges
    || activeDirtyState.coverNeedsRefresh
  const boundaryRerenderMessage = activePendingOperation === 'boundary' && activeClip.renderStatus === 'Rendering'
    ? t(locale, 'boundaryRerenderStarted')
    : null
  const subtitleRerenderMessage = activePendingOperation === 'subtitles' && activeClip.renderStatus === 'Rendering'
    ? t(locale, 'subtitleRerenderStarted')
    : null
  const coverRerenderMessage = activePendingOperation === 'cover' && activeClip.renderStatus === 'Rendering'
    ? t(locale, 'coverRerenderStarted')
    : null
  const subtitleCueTextareaLabel = (index: number) => t(locale, 'subtitleCueTextareaLabel', { index })
  const translatedSubtitleCueTextareaLabel = (index: number) => t(locale, 'translatedSubtitleCueTextareaLabel', { index })
  const translatedSubtitleByIndex = new Map(
    activeClip.translatedSubtitleSegments.map((segment) => [segment.index, segment]),
  )

  function updateSubtitleSegmentText(index: number, text: string, track: 'original' | 'translated' = 'original') {
    if (!activeClip) return
    updateClip(activeClip.id, (clip) => {
      const currentSegments = track === 'translated' ? clip.translatedSubtitleSegments : clip.subtitleSegments
      const nextSegments = currentSegments.map((segment) => (
        segment.index === index
          ? { ...segment, text }
          : segment
      ))
      if (track === 'translated') {
        return {
          ...clip,
          translatedSubtitleSegments: nextSegments,
          translatedSubtitleText: subtitleSegmentsToText(nextSegments),
          hasTranslatedSubtitles: nextSegments.length > 0,
        }
      }
      return {
        ...clip,
        subtitleSegments: nextSegments,
        subtitleText: subtitleSegmentsToText(nextSegments),
      }
    })
  }

  function getQueueButtonLabel(action: 'boundary' | 'subtitles' | 'cover'): string {
    if (activePendingOperation === action) {
      return activeClip.renderStatus === 'Rendering'
        ? t(locale, 'rerenderInProgress', { operation: getOperationLabel(action) })
        : t(locale, 'queueing')
    }
    return t(locale, 'queueRerender', { operation: getOperationLabel(action) })
  }

  return (
    <div className="shell" data-theme={theme}>
      <header className="shell__header">
        <div>
          <p className="eyebrow">{t(locale, 'openClipEditor')}</p>
          <h1>{draftProject.projectName}</h1>
          <p className="shell__subhead">{t(locale, 'projectLabel')} <code>{draftProject.projectId}</code> · {t(locale, 'sourceLabel')} <code>{draftProject.sourceLabel}</code></p>
          <p className="muted">{resolveText(statusMessage)}</p>
        </div>
        <div className="shell__header-side">
          <div className="shell__header-controls">
            <div className="language-switch" role="group" aria-label={t(locale, 'languageSwitchLabel')}>
              <button type="button" className={`language-switch__option ${locale === 'en' ? 'language-switch__option--active' : ''}`} onClick={() => setLocale('en')}>EN</button>
              <button type="button" className={`language-switch__option ${locale === 'zh' ? 'language-switch__option--active' : ''}`} onClick={() => setLocale('zh')}>中文</button>
            </div>
            <button
              type="button"
              className="theme-toggle"
              aria-label={theme === 'dark' ? t(locale, 'switchToLightTheme') : t(locale, 'switchToDarkTheme')}
              onClick={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
            >
              {theme === 'dark' ? t(locale, 'lightTheme') : t(locale, 'darkTheme')}
            </button>
            <button
              type="button"
              className={`status-chip status-chip--${diagnosticsStatusKind}`}
              aria-label={t(locale, 'openDiagnosticsDrawer')}
              onClick={() => setDiagnosticsOpen(true)}
            >
              <span className="status-chip__label">{t(locale, 'diagnosticsLabel')}</span>
              <strong>{diagnosticsStatus}</strong>
            </button>
          </div>
          <div className="summary-strip" aria-label={t(locale, 'projectSummary')}>
            <article><span>{draftProject.clips.length}</span><p>{t(locale, 'clipsInBrowser')}</p></article>
            <article><span>{dirtyClipCount}</span><p>{t(locale, 'dirtyClips')}</p></article>
            <article><span>{formatTimestamp(draftProject.totalDuration)}</span><p>{t(locale, 'sourceTimeline')}</p></article>
          </div>
        </div>
      </header>

      <main className="workspace-grid">
        <aside className="panel clip-browser" aria-label={t(locale, 'clipBrowser')}>
          <div className="panel__header"><div><p className="eyebrow">{t(locale, 'browserEyebrow')}</p><h2>{t(locale, 'clipsTitle')}</h2></div><span className="pill pill--neutral">{t(locale, 'oneActiveClipAtATime')}</span></div>
          <div className="clip-list" role="list">
            {draftProject.clips.map((clip) => {
              const dirtyState = getDirtyState(savedClipMap.get(clip.id) ?? clip, clip)
              const isActive = clip.id === activeClip.id
              return (
                <button key={clip.id} type="button" className={`clip-card ${isActive ? 'clip-card--active' : ''}`} onClick={() => setActiveClipId(clip.id)}>
                  <div className="clip-card__topline"><span className="clip-card__index">#{clip.order}</span><span className={`pill ${dirtyState.hasChanges || clip.coverDirty ? 'pill--warning' : 'pill--neutral'}`}>{getDirtyStateLabel({ ...dirtyState, coverNeedsRefresh: Boolean(clip.coverDirty) || dirtyState.coverNeedsRefresh })}</span></div>
                  <strong>{clip.title}</strong>
                  <p>{clip.sourcePart ? `${clip.sourcePart} · ${t(locale, 'localLabel')} ${clip.localTimeRange}` : clip.localTimeRange}</p>
                  <div className="clip-card__meta"><span>{formatTimestamp(clip.start)} → {formatTimestamp(clip.end)}</span><span>{getRenderStatusLabel(clip.renderStatus)}</span></div>
                </button>
              )
            })}
          </div>
        </aside>

        <section className="panel editor-pane" aria-label={t(locale, 'activeClipEditor')}>
          <div className="panel__header panel__header--stacked">
            <div>
              <p className="eyebrow">{t(locale, 'activeWorkspaceEyebrow')}</p>
              <h2>{activeClip.title}</h2>
              <p className="muted">{activeClip.sourcePart || t(locale, 'singleSource')} · {t(locale, 'localLabel')} {activeClip.localTimeRange} · {t(locale, 'lastUpdate')} {activeClip.updatedAt}</p>
            </div>
            {shouldShowStatusBanner ? (
              <div className="dirty-banner" data-state={activeDirtyState.hasChanges || activeDirtyState.coverNeedsRefresh ? 'dirty' : 'clean'}>
                <strong>
                  {loading
                    ? t(locale, 'loadingProject')
                    : activeClip.renderStatus === 'Recoverable'
                      ? t(locale, 'recoverableRerenderDetected')
                      : activeClip.renderStatus === 'Error'
                        ? t(locale, 'editorActionFailed')
                        : t(locale, 'dirtyStateDetected')}
                </strong>
                <span>
                  {activeClip.lastError
                    ? activeClip.lastError
                    : activeClip.renderStatus === 'Recoverable'
                      ? t(locale, 'resumeInterruptedHint')
                      : activeDirtyState.hasChanges || activeDirtyState.coverNeedsRefresh
                        ? t(locale, 'queueSpecificRerenderHint')
                        : ''}
                </span>
              </div>
            ) : null}
          </div>

          <section className="editor-section">
            <div className="editor-section__header">
              <div><h3>{t(locale, 'timelinePreview')}</h3><p>{t(locale, 'timelinePreviewDescription')}</p></div>
              <span className="pill pill--info">{t(locale, 'previewAndRerender')}</span>
            </div>
            <div className="timeline-shell">
              <div className="timeline-shell__track" aria-hidden="true" ref={timelineTrackRef}>
                {draftProject.clips.map((clip) => {
                  if (clip.end < timelineViewStart || clip.start > timelineViewEnd) return null
                  const left = timelineViewDuration > 0 ? ((clip.start - timelineViewStart) / timelineViewDuration) * 100 : 0
                  const width = timelineViewDuration > 0 ? ((clip.end - clip.start) / timelineViewDuration) * 100 : 0
                  return (
                    <div
                      key={clip.id}
                      className={`timeline-shell__clip ${clip.id === activeClip.id ? 'timeline-shell__clip--active' : ''}`}
                      style={{ left: `${left}%`, width: `${width}%` }}
                    >
                      <span>#{clip.order}</span>
                    </div>
                  )
                })}
                <div className="timeline-shell__window" style={{ left: `${timelineStart}%`, width: `${timelineWidth}%` }}>
                  <button type="button" className="timeline-shell__handle timeline-shell__handle--start" aria-label={t(locale, 'dragStartHandle')} onPointerDown={() => setDragHandle('start')} />
                  <span>#{activeClip.order}</span>
                  <button type="button" className="timeline-shell__handle timeline-shell__handle--end" aria-label={t(locale, 'dragEndHandle')} onPointerDown={() => setDragHandle('end')} />
                </div>
              </div>
              <div className="timeline-shell__controls">
                <label><span>{t(locale, 'start')}</span><input aria-label={t(locale, 'clipStart')} type="range" min={timelineViewStart} max={Math.max(timelineViewEnd - 0.1, timelineViewStart + 0.1)} step={0.1} value={activeClip.start} disabled={loading} onChange={(event) => updateBoundsLocally(Number(event.currentTarget.value), activeClip.end)} /><output>{formatTimestamp(activeClip.start)}</output></label>
                <label><span>{t(locale, 'end')}</span><input aria-label={t(locale, 'clipEnd')} type="range" min={Math.min(timelineViewStart + 0.1, timelineViewEnd)} max={Math.max(timelineViewEnd, timelineViewStart + 0.1)} step={0.1} value={activeClip.end} disabled={loading} onChange={(event) => updateBoundsLocally(activeClip.start, Number(event.currentTarget.value))} /><output>{formatTimestamp(activeClip.end)}</output></label>
              </div>
              <div className="timeline-shell__summary">
                <article><span>{t(locale, 'candidateClip')}</span><strong>{formatTimestamp(activeClip.start)} → {formatTimestamp(activeClip.end)}</strong></article>
                <article><span>{t(locale, 'duration')}</span><strong>{formatTimestamp(outputDuration)}</strong></article>
                <article><span>{t(locale, 'partLocalDebug')}</span><strong>{formatTimestamp(activeClip.localStart)} → {formatTimestamp(activeClip.localEnd)}</strong></article>
                <article><span>{locale === 'zh' ? '可视范围' : 'Visible range'}</span><strong>{formatTimestamp(timelineViewStart)} → {formatTimestamp(timelineViewEnd)}</strong></article>
                <article><span>{t(locale, 'coverState')}</span><strong>{activeDirtyState.coverNeedsRefresh ? t(locale, 'needsRerender') : t(locale, 'currentAssetsUsable')}</strong></article>
              </div>
              {previewSourceUrl ? (
                <div className="timeline-shell__preview">
                  <video ref={previewVideoRef} controls preload="metadata" src={previewSourceUrl} />
                  <p className="muted">{t(locale, 'previewPlaysSelectedWindow')}</p>
                </div>
              ) : (
                <p className="muted">{t(locale, 'sourceVideoPreviewUnavailable')}</p>
              )}
              <div className="timeline-shell__actions">
                <div className="timeline-shell__primary-actions">
                  <button type="button" className="action-button action-button--secondary" disabled={loading || !activeDirtyState.boundaryDirty || Boolean(boundaryRerenderMessage)} onClick={() => void handleQueue('boundary')}>{getQueueButtonLabel('boundary')}</button>
                  <label className="timeline-speed-control">
                    <span>{t(locale, 'playbackSpeed')}</span>
                    <select
                      aria-label={t(locale, 'playbackSpeed')}
                      value={String(activeClip.speed)}
                      disabled={loading}
                      onChange={(event) => {
                        const speed = Number(event.currentTarget.value)
                        updateClip(activeClip.id, (clip) => ({
                          ...clip,
                          speed,
                          coverDirty: true,
                        }))
                      }}
                    >
                      {speedOptions.map((speed) => <option key={speed} value={String(speed)}>{formatSpeedLabel(speed)}</option>)}
                    </select>
                  </label>
                </div>
                {activeClip.renderStatus === 'Recoverable' ? (
                  <button type="button" className="action-button action-button--secondary" disabled={loading} onClick={() => void handleResume()}>{t(locale, 'resumeRerender')}</button>
                ) : null}
              </div>
              {boundaryRerenderMessage ? <p className="rerender-status">{boundaryRerenderMessage}</p> : null}
            </div>
          </section>

          <section className="editor-section">
            <div className="editor-section__header">
              <div><h3>{t(locale, 'subtitleEditor')}</h3><p>{t(locale, 'subtitleEditorDescription')}</p></div>
              <span className={`pill ${activeDirtyState.subtitlesDirty ? 'pill--warning' : 'pill--neutral'}`}>{activeDirtyState.subtitlesDirty ? t(locale, 'dirty') : t(locale, 'clean')}</span>
            </div>
            <div className="subtitle-editor-layout">
              <article className="subtitle-editor-layout__controls">
                <div className="field">
                  <span>{activeClip.hasTranslatedSubtitles ? t(locale, 'translatedSubtitleTrack') : t(locale, 'subtitleOverride')}</span>
                  <div className={`subtitle-cue-table ${activeClip.hasTranslatedSubtitles ? 'subtitle-cue-table--bilingual' : ''}`} aria-label={t(locale, 'subtitleOverride')}>
                    <div className="subtitle-cue-table__header" aria-hidden="true">
                      <span>{t(locale, 'timeRange')}</span>
                      <span>{t(locale, 'subtitleOverride')}</span>
                      {activeClip.hasTranslatedSubtitles ? <span>{t(locale, 'translatedSubtitleTrack')}</span> : null}
                    </div>
                    <div className="subtitle-segment-list">
                      {activeClip.subtitleSegments.length > 0 ? activeClip.subtitleSegments.map((segment: SubtitleSegmentDraft) => {
                        const translatedSegment = translatedSubtitleByIndex.get(segment.index)
                        return (
                          <div key={`${segment.index}-${segment.startTime}-${segment.endTime}`} className="subtitle-segment-card subtitle-segment-card--table">
                            <code className="subtitle-segment-card__time">{segment.startTime} → {segment.endTime}</code>
                            <textarea
                              aria-label={subtitleCueTextareaLabel(segment.index)}
                              value={segment.text}
                              disabled={loading}
                              onChange={(event) => updateSubtitleSegmentText(segment.index, event.currentTarget.value, 'original')}
                              rows={subtitleTextareaRows(segment.text)}
                            />
                            {activeClip.hasTranslatedSubtitles ? (
                              translatedSegment ? (
                                <textarea
                                  aria-label={translatedSubtitleCueTextareaLabel(segment.index)}
                                  value={translatedSegment.text}
                                  disabled={loading}
                                  onChange={(event) => updateSubtitleSegmentText(segment.index, event.currentTarget.value, 'translated')}
                                  rows={subtitleTextareaRows(translatedSegment.text)}
                                />
                              ) : (
                                <p className="muted subtitle-segment-card__missing">{t(locale, 'noTranslatedSubtitleSegmentsAvailable')}</p>
                              )
                            ) : null}
                          </div>
                        )
                      }) : (
                        <p className="muted">{t(locale, 'noSubtitleSegmentsAvailable')}</p>
                      )}
                    </div>
                  </div>
                </div>
                <button type="button" className="action-button action-button--secondary" disabled={loading || !activeDirtyState.subtitlesDirty || Boolean(subtitleRerenderMessage)} onClick={() => void handleQueue('subtitles')}>{getQueueButtonLabel('subtitles')}</button>
                {subtitleRerenderMessage ? <p className="rerender-status">{subtitleRerenderMessage}</p> : null}
              </article>
              <article className="subtitle-preview-panel" aria-label={t(locale, 'postProcessedPreview')}>
                <div className="subtitle-preview-panel__header">
                  <div><h4>{t(locale, 'postProcessedPreview')}</h4><p>{t(locale, 'postProcessedPreviewDescription')}</p></div>
                  <span className={`pill ${subtitlePreviewUrl ? 'pill--info' : 'pill--neutral'}`}>{subtitlePreviewUrl ? t(locale, 'available') : t(locale, 'unavailable')}</span>
                </div>
                {subtitlePreviewUrl ? (
                  <div className="subtitle-preview-panel__media">
                    <video ref={subtitlePreviewVideoRef} key={subtitlePreviewUrl} controls preload="auto" src={subtitlePreviewUrl} />
                    <p className="muted">{t(locale, 'latestRenderedSubtitleOutput')}</p>
                  </div>
                ) : (
                  <div className="subtitle-preview-panel__empty">
                    <strong>{t(locale, 'postProcessedPreviewUnavailable')}</strong>
                    <p>{t(locale, 'noRenderedClipAvailable')}</p>
                  </div>
                )}
              </article>
            </div>
          </section>

          <section className="editor-section">
            <div className="editor-section__header"><div><h3>{t(locale, 'coverTitleEditor')}</h3><p>{t(locale, 'coverTitleEditorDescription')}</p></div><span className={`pill ${activeDirtyState.coverNeedsRefresh ? 'pill--warning' : 'pill--neutral'}`}>{activeDirtyState.coverNeedsRefresh ? t(locale, 'needsRefresh') : t(locale, 'upToDate')}</span></div>
            <label className="field"><span>{t(locale, 'coverTitle')}</span><input aria-label={t(locale, 'coverTitle')} value={activeClip.coverTitle} disabled={loading} onChange={(event) => {
              const { value } = event.currentTarget
              updateClip(activeClip.id, (clip) => ({ ...clip, coverTitle: value, coverDirty: true }))
            }} /></label>
            <div className="cover-preview">
              <article className="cover-preview-card cover-preview-card--horizontal" aria-label={t(locale, 'horizontalCoverPreview')}>
                <div className="cover-preview-card__header"><span>{t(locale, 'horizontal')}</span></div>
                {horizontalCoverPreviewUrl ? (
                  <img src={horizontalCoverPreviewUrl} alt={t(locale, 'horizontalCoverAlt', { title: activeClip.coverTitle })} className="cover-preview-card__image cover-preview-card__image--horizontal" />
                ) : (
                  <div className="cover-preview-card__empty">
                    <strong>{t(locale, 'horizontalCoverUnavailable')}</strong>
                    <p>{t(locale, 'queueHorizontalCoverRerenderHint')}</p>
                  </div>
                )}
                <p className="cover-preview-card__caption">{activeClip.coverTitle}</p>
              </article>
              <article className="cover-preview-card cover-preview-card--vertical" aria-label={t(locale, 'verticalCoverPreview')}>
                <div className="cover-preview-card__header"><span>{t(locale, 'vertical')}</span></div>
                {verticalCoverPreviewUrl ? (
                  <img src={verticalCoverPreviewUrl} alt={t(locale, 'verticalCoverAlt', { title: activeClip.coverTitle })} className="cover-preview-card__image cover-preview-card__image--vertical" />
                ) : (
                  <div className="cover-preview-card__empty">
                    <strong>{t(locale, 'verticalCoverUnavailable')}</strong>
                    <p>{t(locale, 'queueVerticalCoverRerenderHint')}</p>
                  </div>
                )}
                <p className="cover-preview-card__caption">{activeClip.coverTitle}</p>
              </article>
            </div>
            <button type="button" className="action-button action-button--secondary" disabled={loading || !activeDirtyState.coverNeedsRefresh || Boolean(coverRerenderMessage)} onClick={() => void handleQueue('cover')}>{getQueueButtonLabel('cover')}</button>
            {coverRerenderMessage ? <p className="rerender-status">{coverRerenderMessage}</p> : null}
          </section>

          <footer className="editor-actions">
            <button type="button" className="action-button action-button--secondary" disabled={loading} onClick={handleResetClip}>{t(locale, 'resetClipDraft')}</button>
          </footer>
        </section>
      </main>

      <div className={`drawer-backdrop ${diagnosticsOpen ? 'drawer-backdrop--visible' : ''}`} onClick={() => setDiagnosticsOpen(false)} aria-hidden={diagnosticsOpen ? 'false' : 'true'} />
      <aside className={`diagnostics-drawer ${diagnosticsOpen ? 'diagnostics-drawer--open' : ''}`} aria-label={t(locale, 'diagnosticsDrawer')}>
        <div className="diagnostics-drawer__header">
          <div>
            <p className="eyebrow">{t(locale, 'runtimeStatus')}</p>
            <h2>{t(locale, 'editorDiagnostics')}</h2>
            <p className="muted">{t(locale, 'diagnosticsDescription')}</p>
          </div>
          <div className="diagnostics-drawer__actions">
            <span className={`pill ${loadError ? 'pill--warning' : 'pill--info'}`}>{diagnosticsStatus}</span>
            <button type="button" className="drawer-close" aria-label={t(locale, 'closeDiagnosticsDrawer')} onClick={() => setDiagnosticsOpen(false)}>{t(locale, 'close')}</button>
          </div>
        </div>
        <section className="integration-list"><h3>{t(locale, 'recentActivity')}</h3><ul>{activityLog.map((entry, index) => <li key={index}>{resolveText(entry)}</li>)}</ul></section>
        <section className="integration-list"><h3>{t(locale, 'expectedServiceContract')}</h3><ul>{editorEndpoints.map((endpoint) => <li key={endpoint}><code>{endpoint}</code></li>)}</ul></section>
      </aside>
    </div>
  )
}

export default App
