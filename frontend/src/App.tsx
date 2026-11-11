import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'

import {
  LayoutDashboard, Images, Database, Layers, ChartBar, Search,
  Bell, Upload, ArrowUpRight, Activity, FolderOpen, Trash2,
  LoaderCircle, Play, CheckCircle2, AlertCircle, ImagePlus,
  Sparkles, RefreshCw,
  MapPin // <-- NAYA ICON ADD KIYA HAI
} from 'lucide-react'

import {
  ResponsiveContainer,
  BarChart, Bar,
  PieChart, Pie, Cell,
  ScatterChart, Scatter, ZAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts'

// --------------------------------------------------
// TYPES
// --------------------------------------------------

interface UploadedImage {
  filename: string
  url: string
}

interface ImagesResponse {
  total_images: number
  images: UploadedImage[]
}

interface ImageAssignment {
  filename: string
  cluster: number
}

interface PCAPoint {
  filename: string
  cluster: number
  x: number
  y: number
}

interface ClusterSummary {
  cluster: number
  image_count: number
  percentage: number
}

interface ClusteringResults {
  algorithm: string
  total_images: number
  total_clusters: number
  cluster_distribution: Record<string, number>
  cluster_summary?: ClusterSummary[]
  silhouette_score: number | null
  davies_bouldin_score: number | null
  inertia: number | null
  image_assignments: ImageAssignment[]
  pca_points?: PCAPoint[]
  explained_variance?: number[]
  total_explained_variance?: number
  execution_time?: number
}

interface ClusteringResponse {
  message: string
  results: ClusteringResults
}

interface ErrorResponse {
  detail?: string
}

// NAYE TYPES: Recognition ke liye
interface RecognizedImage { filename: string; url: string; confidence: number; }
interface LandmarkGroup { landmark_name: string; image_count: number; images: RecognizedImage[]; }
interface RecognitionResponse { message: string; total_groups: number; groups: LandmarkGroup[]; }

// --------------------------------------------------
// CONSTANTS
// --------------------------------------------------

const API_URL = 'http://127.0.0.1:8001'

const CHART_COLORS = [
  '#547566',
  '#8EAA98',
  '#C19A6B',
  '#7895A8',
  '#B58A9A',
  '#A2A66C',
  '#9A8BC0',
  '#D19A72',
]

const menuItems = [
  { name: 'Dashboard', icon: LayoutDashboard },
  { name: 'Dataset Manager', icon: Images },
  { name: 'Feature Extraction', icon: Database },
  { name: 'Clustering Studio', icon: Layers },
  { name: 'Landmark Recognition', icon: MapPin }, // <-- NAYA MENU ITEM
  { name: 'Evaluation', icon: ChartBar },
]

// --------------------------------------------------
// MAIN APP
// --------------------------------------------------

function App() {
  const [backendStatus, setBackendStatus] = useState<
    'checking' | 'healthy' | 'offline'
  >('checking')

  const [images, setImages] = useState<UploadedImage[]>([])
  const [totalImages, setTotalImages] = useState(0)

  const [uploading, setUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')

  const [imagesLoading, setImagesLoading] = useState(true)
  const [deletingFilename, setDeletingFilename] = useState<string | null>(null)

  const [algorithm, setAlgorithm] = useState('kmeans')
  const [nClusters, setNClusters] = useState(2)
  const [clustering, setClustering] = useState(false)
  const [clusteringMessage, setClusteringMessage] = useState('')

  const [clusteringResults, setClusteringResults] =
    useState<ClusteringResults | null>(null)

  const [experimentCount, setExperimentCount] = useState(0)

  // NAYE STATES: Recognition ke liye
  const [recognizing, setRecognizing] = useState(false)
  const [recognitionResults, setRecognitionResults] = useState<LandmarkGroup[] | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  // --------------------------------------------------
  // LOAD IMAGES
  // --------------------------------------------------

  const loadImages = useCallback(async () => {
    try {
      setImagesLoading(true)

      const response = await fetch(`${API_URL}/images`)

      if (!response.ok) {
        throw new Error('Unable to load images')
      }

      const data: ImagesResponse = await response.json()

      setImages(data.images || [])
      setTotalImages(data.total_images || 0)
    } catch (error) {
      console.error('Image loading error:', error)
    } finally {
      setImagesLoading(false)
    }
  }, [])

  // --------------------------------------------------
  // CHECK BACKEND
  // --------------------------------------------------

  const checkBackend = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/health`)

      if (!response.ok) {
        throw new Error('Backend is not healthy')
      }

      setBackendStatus('healthy')
    } catch (error) {
      console.error('Backend connection error:', error)
      setBackendStatus('offline')
    }
  }, [])

  useEffect(() => {
    checkBackend()
    loadImages()
  }, [checkBackend, loadImages])

  // --------------------------------------------------
  // UPLOAD IMAGES
  // --------------------------------------------------

  const handleImageUpload = async (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const selectedFiles = event.target.files

    if (!selectedFiles || selectedFiles.length === 0) return

    try {
      setUploading(true)
      setUploadMessage('Uploading images...')

      const formData = new FormData()

      Array.from(selectedFiles).forEach((file) => {
        formData.append('files', file)
      })

      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Image upload failed')
      }

      setUploadMessage('Images uploaded successfully!')

      // New upload ke baad old results clear karte hain.
      setClusteringResults(null)
      setRecognitionResults(null) // Reset Recognition
      await loadImages()
    } catch (error) {
      setUploadMessage(
        error instanceof Error ? error.message : 'Upload failed',
      )
    } finally {
      setUploading(false)

      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  // --------------------------------------------------
  // DELETE IMAGE
  // --------------------------------------------------

  const handleDeleteImage = async (filename: string) => {
    const confirmDelete = window.confirm(
      'Are you sure you want to delete this image?',
    )

    if (!confirmDelete) return

    try {
      setDeletingFilename(filename)

      const response = await fetch(
        `${API_URL}/images/${encodeURIComponent(filename)}`,
        { method: 'DELETE' },
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Unable to delete image')
      }

      setClusteringResults(null)
      setRecognitionResults(null)
      await loadImages()
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Delete failed')
    } finally {
      setDeletingFilename(null)
    }
  }

  // --------------------------------------------------
  // RUN CLUSTERING
  // --------------------------------------------------

  const handleRunClustering = async () => {
    if (totalImages < 2) {
      setClusteringMessage('Please upload at least 2 images first.')
      return
    }

    if (nClusters < 2 || nClusters >= totalImages) {
      setClusteringMessage(
        `Choose clusters between 2 and ${totalImages - 1}.`,
      )
      return
    }

    try {
      setClustering(true)
      setClusteringMessage('Clustering is running...')

      const params = new URLSearchParams({
        algorithm,
        n_clusters: nClusters.toString(),
      })

      const response = await fetch(
        `${API_URL}/cluster?${params.toString()}`,
        { method: 'POST' },
      )

      const data: ClusteringResponse | ErrorResponse =
        await response.json()

      if (!response.ok) {
        throw new Error(
          'detail' in data
            ? data.detail || 'Clustering failed'
            : 'Clustering failed',
        )
      }

      if ('results' in data) {
        setClusteringResults(data.results)
        setExperimentCount((previous) => previous + 1)
        setClusteringMessage(data.message)
      }
    } catch (error) {
      setClusteringMessage(
        error instanceof Error ? error.message : 'Clustering failed',
      )
    } finally {
      setClustering(false)
    }
  }

  // --------------------------------------------------
  // NAYA FUNCTION: RUN RECOGNITION
  // --------------------------------------------------

  const handleRunRecognition = async () => {
    if (totalImages === 0) return alert('Please upload images first.')
    try {
      setRecognizing(true)
      const response = await fetch(`${API_URL}/recognize`)
      const data: RecognitionResponse = await response.json()
      if (!response.ok) throw new Error('Recognition failed')
      setRecognitionResults(data.groups)
    } catch (error) {
      alert('Error during recognition: ' + error)
    } finally {
      setRecognizing(false)
    }
  }

  // --------------------------------------------------
  // FORMAT ALGORITHM NAME
  // --------------------------------------------------

  const formatAlgorithm = (name: string) => {
    const names: Record<string, string> = {
      kmeans: 'K-Means',
      kmedians: 'K-Medians',
      agglomerative: 'Agglomerative',
    }

    return names[name] || name
  }

  // --------------------------------------------------
  // GRAPH DATA
  // --------------------------------------------------

  const clusterChartData = clusteringResults
    ? Object.entries(clusteringResults.cluster_distribution).map(
        ([clusterId, count]) => ({
          name: `Cluster ${Number(clusterId) + 1}`,
          images: count,
          value: count,
          percentage:
            clusteringResults.total_images > 0
              ? (count / clusteringResults.total_images) * 100
              : 0,
        }),
      )
    : []

  const pcaSeries = clusteringResults
    ? Array.from(
        { length: clusteringResults.total_clusters },
        (_, clusterId) => ({
          clusterId,
          name: `Cluster ${clusterId + 1}`,
          points: (clusteringResults.pca_points || []).filter(
            (point) => point.cluster === clusterId,
          ),
        }),
      ).filter((series) => series.points.length > 0)
    : []

  const varianceData = (clusteringResults?.explained_variance || []).map(
    (value, index) => ({
      component: `PC ${index + 1}`,
      variance: Number((value * 100).toFixed(2)),
    }),
  )

  // --------------------------------------------------
  // DASHBOARD STATS
  // --------------------------------------------------

  const stats = [
    {
      title: 'Total Images',
      value: totalImages.toString(),
      subtitle: 'Uploaded images',
      icon: Images,
    },
    {
      title: 'Experiments',
      value: experimentCount.toString(),
      subtitle: 'Successful runs',
      icon: Activity,
    },
    {
      title: 'Clusters Created',
      value: clusteringResults
        ? clusteringResults.total_clusters.toString()
        : '0',
      subtitle: 'Latest experiment',
      icon: Layers,
    },
    {
      title: 'ML Engine',
      value: backendStatus === 'healthy' ? 'Ready' : 'Offline',
      subtitle: 'Backend connection',
      icon: Sparkles,
    },
  ]

  // --------------------------------------------------
  // UI
  // --------------------------------------------------

  return (
    <div className="min-h-screen bg-[#F7F8F5] text-[#25352F]">

      {/* SIDEBAR */}

      <aside className="fixed left-0 top-0 z-20 hidden h-screen w-64 border-r border-[#E7EBE6] bg-white px-5 py-7 lg:flex lg:flex-col">
        <div className="mb-10 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#E8EEE9] text-[#547566]">
            <Layers size={23} />
          </div>

          <div>
            <h1 className="text-lg font-bold tracking-tight">Landmark AI</h1>
            <p className="text-xs text-[#87938C]">Studio</p>
          </div>
        </div>

        <p className="mb-4 px-3 text-xs font-semibold uppercase tracking-[0.15em] text-[#A0AAA3]">
          Workspace
        </p>

        <nav className="space-y-2">
          {menuItems.map((item, index) => {
            const Icon = item.icon

            return (
              <button
                key={item.name}
                onClick={() => {
                  const sectionIds = [
                    'dashboard',
                    'dataset',
                    'feature-extraction',
                    'clustering',
                    'recognition', // <-- NAYA ID
                    'evaluation',
                  ]

                  document
                    .getElementById(sectionIds[index])
                    ?.scrollIntoView({ behavior: 'smooth' })
                }}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition ${
                  index === 0
                    ? 'bg-[#E8EEE9] font-semibold text-[#456657]'
                    : 'text-[#7B8981] hover:bg-[#F5F7F4] hover:text-[#25352F]'
                }`}
              >
                <Icon size={19} />
                {item.name}
              </button>
            )
          })}
        </nav>

        <div className="mt-auto rounded-2xl bg-[#F3F6F2] p-4">
          <div className="mb-2 flex items-center gap-2">
            <Sparkles size={17} className="text-[#6F8F83]" />
            <span className="text-sm font-semibold">AI Workspace</span>
          </div>

          <p className="text-xs leading-5 text-[#849087]">
            Group landmark images using visual feature similarity.
          </p>
        </div>
      </aside>

      {/* MAIN CONTENT */}

      <main className="min-h-screen lg:ml-64">

        {/* HEADER */}

        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-[#E7EBE6] bg-[#F7F8F5]/95 px-5 py-4 backdrop-blur-md md:px-8">
          <div>
            <p className="text-xs text-[#89968D]">Workspace / Overview</p>
            <h2 className="mt-1 text-xl font-bold tracking-tight">Dashboard</h2>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 rounded-xl border border-[#E7EBE6] bg-white px-3 py-2 sm:flex">
              <Search size={16} className="text-[#9AA69E]" />
              <span className="text-xs text-[#9AA69E]">Search workspace</span>
            </div>

            <button className="rounded-xl border border-[#E7EBE6] bg-white p-2.5 text-[#718077]">
              <Bell size={18} />
            </button>

            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#DCE8DF] text-sm font-bold text-[#547566]">
              VK
            </div>
          </div>
        </header>

        <div className="space-y-8 px-5 py-7 md:px-8">

          {/* DASHBOARD */}

          <section id="dashboard" className="scroll-mt-28">
            <div className="flex flex-col justify-between gap-5 md:flex-row md:items-center">
              <div>
                <p className="mb-2 text-sm font-medium text-[#789083]">
                  Welcome to your workspace
                </p>

                <h1 className="text-3xl font-bold tracking-tight md:text-4xl">
                  Landmark Photo Studio
                </h1>

                <p className="mt-3 max-w-2xl text-sm leading-6 text-[#849087]">
                  Organize and analyze landmark photographs using feature
                  extraction, machine learning and clustering.
                </p>
              </div>

              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className="flex items-center justify-center gap-2 rounded-xl bg-[#547566] px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-[#456657] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {uploading ? (
                  <LoaderCircle size={18} className="animate-spin" />
                ) : (
                  <Upload size={18} />
                )}
                {uploading ? 'Uploading...' : 'Upload Images'}
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp"
                multiple
                className="hidden"
                onChange={handleImageUpload}
              />
            </div>

            {uploadMessage && (
              <div className="mt-4 flex items-center gap-2 rounded-xl border border-[#E3EAE3] bg-white px-4 py-3 text-sm text-[#547566]">
                <CheckCircle2 size={17} />
                {uploadMessage}
              </div>
            )}
          </section>

          {/* STATISTICS */}

          <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {stats.map((stat) => {
              const Icon = stat.icon

              return (
                <div
                  key={stat.title}
                  className="rounded-2xl border border-[#E8ECE7] bg-white p-5 shadow-sm"
                >
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-[#89968D]">{stat.title}</p>
                    <div className="rounded-xl bg-[#EFF4EF] p-2.5 text-[#648273]">
                      <Icon size={19} />
                    </div>
                  </div>

                  <h3 className="mt-4 text-2xl font-bold">{stat.value}</h3>
                  <p className="mt-1 text-xs text-[#9AA69E]">{stat.subtitle}</p>
                </div>
              )
            })}
          </section>

          {/* BACKEND STATUS */}

          <section className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[#E8ECE7] bg-white px-5 py-4">
            <div className="flex items-center gap-3">
              <div
                className={`h-2.5 w-2.5 rounded-full ${
                  backendStatus === 'healthy'
                    ? 'bg-green-500'
                    : backendStatus === 'offline'
                      ? 'bg-red-500'
                      : 'animate-pulse bg-yellow-500'
                }`}
              />

              <div>
                <p className="text-sm font-semibold">Backend Connection</p>
                <p className="text-xs text-[#89968D]">
                  {backendStatus === 'healthy'
                    ? 'FastAPI server is connected'
                    : backendStatus === 'offline'
                      ? 'Unable to connect to backend'
                      : 'Checking server connection...'}
                </p>
              </div>
            </div>

            <button
              onClick={checkBackend}
              className="flex items-center gap-2 rounded-lg border border-[#E7EBE6] px-3 py-2 text-xs font-medium text-[#64776B] hover:bg-[#F5F7F4]"
            >
              <RefreshCw size={14} />
              Check again
            </button>
          </section>

          {/* DATASET MANAGER */}

          <section id="dataset" className="scroll-mt-28">
            <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#8A9A8E]">
                  Your data
                </p>
                <h2 className="mt-1 text-xl font-bold">Dataset Manager</h2>
                <p className="mt-1 text-sm text-[#89968D]">
                  Manage your uploaded landmark photographs.
                </p>
              </div>

              <span className="rounded-full bg-[#E8EEE9] px-3 py-1.5 text-xs font-semibold text-[#547566]">
                {totalImages} Images
              </span>
            </div>

            {imagesLoading ? (
              <div className="flex items-center justify-center rounded-2xl border border-[#E8ECE7] bg-white py-16">
                <LoaderCircle size={25} className="animate-spin text-[#6F8F83]" />
                <span className="ml-3 text-sm text-[#89968D]">Loading images...</span>
              </div>
            ) : images.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-[#CCD8CE] bg-white px-5 py-16 text-center">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-[#EFF4EF] text-[#6F8F83]">
                  <ImagePlus size={26} />
                </div>

                <h3 className="mt-4 font-semibold">No images uploaded yet</h3>
                <p className="mt-2 text-sm text-[#89968D]">
                  Upload landmark photographs to start your project.
                </p>

                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-5 rounded-xl bg-[#547566] px-4 py-2.5 text-sm font-semibold text-white hover:bg-[#456657]"
                >
                  Upload your first image
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
                {images.map((image) => (
                  <div
                    key={image.filename}
                    className="group overflow-hidden rounded-2xl border border-[#E8ECE7] bg-white shadow-sm"
                  >
                    <div className="relative aspect-[4/3] bg-[#F0F3EF]">
                      <img
                        src={image.url}
                        alt={image.filename}
                        className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                        onError={(event) => {
                          event.currentTarget.style.display = 'none'
                        }}
                      />

                      <button
                        onClick={() => handleDeleteImage(image.filename)}
                        disabled={deletingFilename === image.filename}
                        title="Delete image"
                        className="absolute right-2 top-2 rounded-lg bg-white/95 p-2 text-red-500 opacity-100 shadow-sm transition hover:bg-red-50 md:opacity-0 md:group-hover:opacity-100"
                      >
                        {deletingFilename === image.filename ? (
                          <LoaderCircle size={16} className="animate-spin" />
                        ) : (
                          <Trash2 size={16} />
                        )}
                      </button>
                    </div>

                    <div className="p-3">
                      <p className="truncate text-xs font-medium text-[#586A5E]" title={image.filename}>
                        {image.filename}
                      </p>
                      <p className="mt-1 text-[11px] text-[#A0AAA3]">Uploaded image</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* FEATURE EXTRACTION */}

          <section
            id="feature-extraction"
            className="scroll-mt-28 rounded-2xl border border-[#E8ECE7] bg-white p-5 md:p-6"
          >
            <div className="flex items-start gap-4">
              <div className="rounded-xl bg-[#EFF4EF] p-3 text-[#648273]">
                <Database size={22} />
              </div>

              <div>
                <h2 className="text-lg font-bold">Feature Extraction</h2>
                <p className="mt-1 max-w-2xl text-sm leading-6 text-[#89968D]">
                  The ML pipeline extracts visual information from every image
                  to create a feature vector for clustering.
                </p>
              </div>
            </div>

            <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="rounded-xl bg-[#F7F8F5] p-4">
                <p className="text-sm font-semibold">ResNet-50</p>
                <p className="mt-1 text-xs text-[#89968D]">2048 deep visual features</p>
              </div>

              <div className="rounded-xl bg-[#F7F8F5] p-4">
                <p className="text-sm font-semibold">RGB Histogram</p>
                <p className="mt-1 text-xs text-[#89968D]">64 colour features</p>
              </div>

              <div className="rounded-xl bg-[#F7F8F5] p-4">
                <p className="text-sm font-semibold">LBP Texture</p>
                <p className="mt-1 text-xs text-[#89968D]">16 texture features</p>
              </div>
            </div>

            <div className="mt-4 flex items-center justify-between rounded-xl border border-[#E8ECE7] px-4 py-3">
              <span className="text-sm text-[#718077]">Total hybrid feature dimensions</span>
              <span className="text-lg font-bold text-[#547566]">2,128</span>
            </div>
          </section>

          {/* CLUSTERING STUDIO */}

          <section id="clustering" className="scroll-mt-28">
            <div className="mb-5">
              <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#8A9A8E]">
                Machine learning
              </p>
              <h2 className="mt-1 text-xl font-bold">Clustering Studio</h2>
              <p className="mt-1 text-sm text-[#89968D]">
                Select an algorithm and group similar landmark photographs.
              </p>
            </div>

            {/* Clustering controls */}

            <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5 shadow-sm md:p-6">
              <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                <div>
                  <label className="mb-2 block text-sm font-semibold">Select Algorithm</label>

                  <select
                    value={algorithm}
                    onChange={(event) => {
                      setAlgorithm(event.target.value)
                      setClusteringResults(null)
                      setClusteringMessage('')
                    }}
                    className="w-full rounded-xl border border-[#DFE6DF] bg-[#FAFBF9] px-4 py-3 text-sm outline-none transition focus:border-[#7F9B8C]"
                  >
                    <option value="kmeans">K-Means</option>
                    <option value="kmedians">K-Medians</option>
                    <option value="agglomerative">Agglomerative Clustering</option>
                  </select>

                  <p className="mt-2 text-xs text-[#9AA69E]">Choose the clustering method.</p>
                </div>

                <div>
                  <label className="mb-2 block text-sm font-semibold">Number of Clusters (K)</label>

                  <input
                    type="number"
                    min={2}
                    max={Math.max(2, totalImages - 1)}
                    value={nClusters}
                    onChange={(event) => {
                      setNClusters(Number(event.target.value))
                      setClusteringResults(null)
                      setClusteringMessage('')
                    }}
                    className="w-full rounded-xl border border-[#DFE6DF] bg-[#FAFBF9] px-4 py-3 text-sm outline-none transition focus:border-[#7F9B8C]"
                  />

                  <p className="mt-2 text-xs text-[#9AA69E]">Select how many groups you want to create.</p>
                </div>
              </div>

              <div className="mt-6 flex flex-col gap-3 border-t border-[#EEF1EC] pt-5 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-xs text-[#89968D]">{totalImages} images available for clustering.</p>

                <button
                  onClick={handleRunClustering}
                  disabled={clustering || totalImages < 2}
                  className="flex items-center justify-center gap-2 rounded-xl bg-[#547566] px-5 py-3 text-sm font-semibold text-white transition hover:bg-[#456657] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {clustering ? (
                    <LoaderCircle size={18} className="animate-spin" />
                  ) : (
                    <Play size={17} />
                  )}
                  {clustering ? 'Processing Images...' : 'Run Clustering'}
                </button>
              </div>

              {clusteringMessage && (
                <div className="mt-4 flex items-start gap-2 rounded-xl border border-[#E4EAE3] bg-[#F5F8F4] p-3 text-sm text-[#547566]">
                  {clustering ? (
                    <LoaderCircle size={17} className="mt-0.5 animate-spin" />
                  ) : clusteringResults ? (
                    <CheckCircle2 size={17} className="mt-0.5" />
                  ) : (
                    <AlertCircle size={17} className="mt-0.5" />
                  )}
                  <span>{clusteringMessage}</span>
                </div>
              )}
            </div>

            {/* RESULTS AND GRAPHS */}

            {clusteringResults && (
              <div className="mt-6 space-y-5">

                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="text-lg font-bold">Clustering Results</h3>
                    <p className="mt-1 text-xs text-[#89968D]">
                      Latest experiment using {formatAlgorithm(clusteringResults.algorithm)}
                    </p>
                  </div>

                  <span className="rounded-full bg-[#E8EEE9] px-3 py-1.5 text-xs font-semibold text-[#547566]">
                    Completed
                  </span>
                </div>

                {/* METRIC CARDS */}

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5">
                    <p className="text-xs text-[#89968D]">Silhouette Score</p>
                    <h3 className="mt-2 text-2xl font-bold text-[#547566]">
                      {clusteringResults.silhouette_score !== null
                        ? clusteringResults.silhouette_score.toFixed(4)
                        : 'N/A'}
                    </h3>
                    <p className="mt-2 text-xs text-[#A0AAA3]">Higher is generally better.</p>
                  </div>

                  <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5">
                    <p className="text-xs text-[#89968D]">Davies-Bouldin Score</p>
                    <h3 className="mt-2 text-2xl font-bold text-[#547566]">
                      {clusteringResults.davies_bouldin_score !== null
                        ? clusteringResults.davies_bouldin_score.toFixed(4)
                        : 'N/A'}
                    </h3>
                    <p className="mt-2 text-xs text-[#A0AAA3]">Lower is generally better.</p>
                  </div>

                  <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5">
                    <p className="text-xs text-[#89968D]">
                      {clusteringResults.algorithm === 'kmeans' ? 'Inertia / WCSS' : 'Inertia / Distance'}
                    </p>
                    <h3 className="mt-2 text-2xl font-bold text-[#547566]">
                      {clusteringResults.inertia !== null
                        ? clusteringResults.inertia.toFixed(2)
                        : 'N/A'}
                    </h3>
                    <p className="mt-2 text-xs text-[#A0AAA3]">Within-cluster distance measure.</p>
                  </div>
                </div>

                {/* CLUSTER DISTRIBUTION BAR CHART */}

                <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5">
                  <h3 className="font-bold">Cluster Distribution — Bar Chart</h3>
                  <p className="mt-1 text-xs text-[#89968D]">
                    Actual number of images assigned to each cluster.
                  </p>

                  <div className="mt-5 h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={clusterChartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E8ECE7" />
                        <XAxis dataKey="name" tick={{ fill: '#718077', fontSize: 12 }} />
                        <YAxis allowDecimals={false} tick={{ fill: '#718077', fontSize: 12 }} />
                        <Tooltip />
                        <Bar dataKey="images" name="Number of Images" fill="#789989" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* PIE CHART */}

                <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5">
                  <h3 className="font-bold">Cluster Distribution — Pie Chart</h3>
                  <p className="mt-1 text-xs text-[#89968D]">
                    Percentage share of images in each cluster.
                  </p>

                  <div className="mt-5 h-80 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={clusterChartData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="48%"
                          outerRadius={105}
                          label={({ name, percent }) =>
                            `${name} (${((percent ?? 0) * 100).toFixed(1)}%)`
                          }
                        >
                          {clusterChartData.map((entry, index) => (
                            <Cell
                              key={entry.name}
                              fill={CHART_COLORS[index % CHART_COLORS.length]}
                            />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* PCA SCATTER PLOT */}

                <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5">
                  <h3 className="font-bold">PCA Visualization — 2D Scatter Plot</h3>
                  <p className="mt-1 text-xs text-[#89968D]">
                    Each dot represents one image. Colours represent assigned clusters.
                  </p>

                  {pcaSeries.length > 0 ? (
                    <div className="mt-5 h-96 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 15, right: 20, bottom: 20, left: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E8ECE7" />
                          <XAxis
                            type="number"
                            dataKey="x"
                            name="PCA Component 1"
                            tick={{ fill: '#718077', fontSize: 11 }}
                            label={{ value: 'PCA Component 1', position: 'insideBottom', offset: -10 }}
                          />
                          <YAxis
                            type="number"
                            dataKey="y"
                            name="PCA Component 2"
                            tick={{ fill: '#718077', fontSize: 11 }}
                          />
                          <ZAxis range={[65, 65]} />
                          <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            formatter={(value) => Number(value).toFixed(3)}
                          />
                          <Legend />

                          {pcaSeries.map((series) => (
                            <Scatter
                              key={series.clusterId}
                              name={series.name}
                              data={series.points}
                              fill={CHART_COLORS[series.clusterId % CHART_COLORS.length]}
                            />
                          ))}
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <p className="mt-5 rounded-xl bg-[#F7F8F5] p-4 text-sm text-[#89968D]">
                      PCA points are not available in the backend response.
                    </p>
                  )}

                  <p className="mt-3 text-xs leading-5 text-[#89968D]">
                    Note: PCA reduces the feature dimensions to two for visualization.
                    Points close together have similar projected feature representations.
                  </p>
                </div>

                {/* EXPLAINED VARIANCE */}

                <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5">
                  <h3 className="font-bold">PCA — Explained Variance</h3>
                  <p className="mt-1 text-xs text-[#89968D]">
                    Percentage of information represented by each principal component.
                  </p>

                  {varianceData.length > 0 ? (
                    <>
                      <div className="mt-5 h-64 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={varianceData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E8ECE7" />
                            <XAxis dataKey="component" tick={{ fill: '#718077', fontSize: 12 }} />
                            <YAxis
                              unit="%"
                              tick={{ fill: '#718077', fontSize: 12 }}
                            />
                            <Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} />
                            <Bar dataKey="variance" name="Explained Variance" fill="#8EAA98" radius={[8, 8, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="mt-4 rounded-xl bg-[#F5F8F4] p-4">
                        <p className="text-sm text-[#718077]">
                          Total variance explained by displayed components
                        </p>
                        <p className="mt-1 text-xl font-bold text-[#547566]">
                          {clusteringResults.total_explained_variance !== undefined
                            ? `${(clusteringResults.total_explained_variance * 100).toFixed(2)}%`
                            : `${varianceData.reduce((sum, item) => sum + item.variance, 0).toFixed(2)}%`}
                        </p>
                      </div>
                    </>
                  ) : (
                    <p className="mt-5 rounded-xl bg-[#F7F8F5] p-4 text-sm text-[#89968D]">
                      Explained variance data is not available.
                    </p>
                  )}
                </div>

                {/* METRIC VISUAL INDICATORS */}

                <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5">
                  <h3 className="font-bold">Model Evaluation Overview</h3>
                  <p className="mt-1 text-xs text-[#89968D]">
                    Visual indicators for the latest clustering experiment.
                  </p>

                  <div className="mt-5 space-y-5">
                    <div>
                      <div className="mb-2 flex justify-between text-sm">
                        <span>Silhouette Score</span>
                        <span className="font-semibold">
                          {clusteringResults.silhouette_score !== null
                            ? clusteringResults.silhouette_score.toFixed(4)
                            : 'N/A'}
                        </span>
                      </div>
                      <div className="h-3 overflow-hidden rounded-full bg-[#EDF1EC]">
                        <div
                          className="h-full rounded-full bg-[#789989]"
                          style={{
                            width: `${Math.max(0, Math.min(100, (clusteringResults.silhouette_score ?? 0) * 100))}%`,
                          }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="mb-2 flex justify-between text-sm">
                        <span>Davies-Bouldin Score</span>
                        <span className="font-semibold">
                          {clusteringResults.davies_bouldin_score !== null
                            ? clusteringResults.davies_bouldin_score.toFixed(4)
                            : 'N/A'}
                        </span>
                      </div>
                      <p className="text-xs leading-5 text-[#89968D]">
                        Lower values generally indicate better separation. This score is
                        displayed as a value, not converted into a percentage.
                      </p>
                    </div>
                  </div>
                </div>

                {/* CLUSTER PROGRESS BARS */}

                <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5">
                  <h3 className="font-bold">Cluster-wise Image Count</h3>

                  <div className="mt-5 space-y-4">
                    {clusterChartData.map((item) => (
                      <div key={item.name}>
                        <div className="mb-2 flex items-center justify-between">
                          <span className="text-sm font-semibold">{item.name}</span>
                          <span className="text-xs text-[#89968D]">
                            {item.images} images ({item.percentage.toFixed(1)}%)
                          </span>
                        </div>

                        <div className="h-3 overflow-hidden rounded-full bg-[#EDF1EC]">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${item.percentage}%`,
                              backgroundColor: CHART_COLORS[
                                clusterChartData.indexOf(item) % CHART_COLORS.length
                              ],
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* CLUSTER-WISE IMAGE GROUPS */}

                <div>
                  <div className="mb-4">
                    <h3 className="text-lg font-bold">Grouped Landmark Images</h3>
                    <p className="mt-1 text-xs text-[#89968D]">
                      Images are displayed according to their assigned visual-similarity
                      cluster. These are not verified landmark identities.
                    </p>
                  </div>

                  <div className="space-y-5">
                    {Object.entries(clusteringResults.cluster_distribution).map(
                      ([clusterId, count]) => {
                        const clusterImages =
                          clusteringResults.image_assignments.filter(
                            (assignment) => assignment.cluster === Number(clusterId),
                          )

                        return (
                          <div
                            key={clusterId}
                            className="rounded-2xl border border-[#E8ECE7] bg-white p-5"
                          >
                            <div className="mb-4 flex items-center justify-between">
                              <div>
                                <h4 className="font-bold">
                                  Cluster {Number(clusterId) + 1}
                                </h4>
                                <p className="mt-1 text-xs text-[#89968D]">
                                  {count} images in this group
                                </p>
                              </div>

                              <div className="rounded-xl bg-[#EFF4EF] p-2.5 text-[#648273]">
                                <FolderOpen size={20} />
                              </div>
                            </div>

                            {clusterImages.length === 0 ? (
                              <p className="text-sm text-[#89968D]">
                                No image assignments found.
                              </p>
                            ) : (
                              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                                {clusterImages.map((assignment) => {
                                  const matchedImage = images.find(
                                    (image) => image.filename === assignment.filename,
                                  )

                                  return (
                                    <div
                                      key={assignment.filename}
                                      className="overflow-hidden rounded-xl border border-[#E8ECE7] bg-[#FAFBF9]"
                                    >
                                      {matchedImage ? (
                                        <img
                                          src={matchedImage.url}
                                          alt={assignment.filename}
                                          className="aspect-[4/3] w-full object-cover"
                                        />
                                      ) : (
                                        <div className="flex aspect-[4/3] items-center justify-center text-[#9AA69E]">
                                          <Images size={25} />
                                        </div>
                                      )}

                                      <p
                                        className="truncate px-2 py-2 text-[11px] text-[#718077]"
                                        title={assignment.filename}
                                      >
                                        {assignment.filename}
                                      </p>
                                    </div>
                                  )
                                })}
                              </div>
                            )}
                          </div>
                        )
                      },
                    )}
                  </div>
                </div>
              </div>
            )}
          </section>

          {/* NAYA SECTION: LANDMARK RECOGNITION */}
          <section id="recognition" className="scroll-mt-28">
            <div className="mb-5">
              <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#8A9A8E]">Semantic Grouping</p>
              <h2 className="mt-1 text-xl font-bold">Landmark Recognition</h2>
              <p className="mt-1 text-sm text-[#89968D]">Identify actual landmark names (Taj Mahal, Qutub Minar) using CLIP AI model.</p>
            </div>

            <div className="rounded-2xl border border-[#E8ECE7] bg-white p-5 shadow-sm md:p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-lg">Recognize Uploaded Photos</h3>
                  <p className="text-sm text-[#89968D] mt-1">Run the AI to group photos by their actual names.</p>
                </div>
                <button onClick={handleRunRecognition} disabled={recognizing || totalImages === 0} className="flex items-center gap-2 rounded-xl bg-[#8EAA98] px-5 py-3 text-sm font-semibold text-white transition hover:bg-[#7B9685] disabled:opacity-50">
                  {recognizing ? <LoaderCircle size={18} className="animate-spin" /> : <MapPin size={18} />}
                  {recognizing ? 'Analyzing Photos...' : 'Recognize Landmarks'}
                </button>
              </div>

              {recognitionResults && (
                <div className="mt-8 space-y-6">
                  {recognitionResults.map((group) => (
                    <div key={group.landmark_name} className="rounded-2xl border border-[#E8ECE7] bg-[#FAFBF9] p-4">
                      <div className="mb-3 flex items-center justify-between border-b border-[#E8ECE7] pb-3">
                        <div className="flex items-center gap-2">
                           <MapPin size={18} className="text-[#547566]" />
                           <h4 className="font-bold text-lg text-[#25352F]">{group.landmark_name}</h4>
                        </div>
                        <span className="rounded-full bg-[#E8EEE9] px-3 py-1 text-xs font-bold text-[#547566]">{group.image_count} Photos</span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                        {group.images.map((img) => (
                          <div key={img.filename} className="overflow-hidden rounded-xl border border-[#E8ECE7] bg-white shadow-sm relative group">
                            <img src={img.url} alt={img.filename} className="aspect-[4/3] w-full object-cover" />
                            <div className="p-2">
                               <p className="truncate text-[11px] text-[#718077]">{img.filename}</p>
                               <p className={`text-[10px] font-bold mt-1 ${img.confidence > 80 ? 'text-green-600' : 'text-yellow-600'}`}>Match: {img.confidence}%</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>

          {/* MODEL EVALUATION */}

          <section
            id="evaluation"
            className="scroll-mt-28 rounded-2xl border border-[#E8ECE7] bg-white p-5 md:p-6"
          >
            <div className="flex items-start gap-3">
              <div className="rounded-xl bg-[#EFF4EF] p-3 text-[#648273]">
                <ChartBar size={22} />
              </div>

              <div>
                <h2 className="text-lg font-bold">Model Evaluation</h2>
                <p className="mt-1 text-sm leading-6 text-[#89968D]">
                  Evaluate clustering quality using Silhouette Score,
                  Davies-Bouldin Index and inertia.
                </p>
              </div>
            </div>

            {clusteringResults ? (
              <div className="mt-5 rounded-xl bg-[#F5F8F4] p-4">
                <p className="text-sm font-semibold text-[#547566]">
                  Latest evaluation is available.
                </p>
                <p className="mt-1 text-xs text-[#89968D]">
                  Current algorithm: {formatAlgorithm(clusteringResults.algorithm)}.
                  Run other algorithms to compare their individual results.
                </p>
              </div>
            ) : (
              <div className="mt-5 rounded-xl bg-[#F7F8F5] p-4">
                <p className="text-sm text-[#89968D]">
                  Run clustering to generate evaluation metrics and graphs.
                </p>
              </div>
            )}
          </section>

          {/* GET STARTED */}

          <section className="overflow-hidden rounded-2xl bg-[#E8EEE9] p-6 md:p-8">
            <div className="flex flex-col justify-between gap-5 md:flex-row md:items-center">
              <div>
                <div className="mb-3 flex items-center gap-2 text-[#648273]">
                  <Sparkles size={18} />
                  <span className="text-xs font-semibold uppercase tracking-[0.15em]">
                    Project workspace
                  </span>
                </div>

                <h2 className="text-xl font-bold">Discover visual similarities</h2>
                <p className="mt-2 max-w-xl text-sm leading-6 text-[#718077]">
                  Extract features from landmark photographs and experiment
                  with different clustering algorithms.
                </p>
              </div>

              <button
                onClick={() =>
                  document.getElementById('clustering')?.scrollIntoView({
                    behavior: 'smooth',
                  })
                }
                className="flex items-center justify-center gap-2 rounded-xl bg-white px-5 py-3 text-sm font-semibold text-[#547566] shadow-sm transition hover:bg-[#F8FAF7]"
              >
                Start Experiment
                <ArrowUpRight size={17} />
              </button>
            </div>
          </section>

          {/* FOOTER */}

          <footer className="flex flex-col justify-between gap-2 border-t border-[#E7EBE6] py-5 text-xs text-[#9AA69E] sm:flex-row">
            <p>Landmark AI Studio · Visual Feature Clustering</p>
            <p>Built with React, FastAPI and Machine Learning</p>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App