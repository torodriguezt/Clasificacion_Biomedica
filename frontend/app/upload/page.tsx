"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import Link from "next/link"
import { Upload, FileText, BarChart3, Brain, Download, Check, X, AlertCircle, CheckCircle, XCircle } from "lucide-react"

type UploadResult = {
  total_processed: number
  predictions: Array<{
    title: string
    predictions: Record<string, number>
    dominant_category: string
    confidence: number
  }>
  data: Array<Record<string, any>>
  evaluation_metrics?: {
    confusion_matrix: {
      matrix: number[][]
      labels: string[]
      format: string
    }
    f1_scores: {
      macro: number
      micro: number
      weighted: number
      samples: number
      per_class: Record<string, number>
    }
    accuracy: {
      subset_accuracy: number
      jaccard_similarity: number
    }
    hamming_loss: number
    precision: number
    recall: number
  }
}

export default function CSVUpload() {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<UploadResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    const files = e.dataTransfer.files
    if (files && files[0]) {
      if (files[0].type === "text/csv" || files[0].name.endsWith('.csv')) {
        setFile(files[0])
        setError(null)
      } else {
        setError("Por favor seleccione un archivo CSV válido")
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      if (selectedFile.type === "text/csv" || selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile)
        setError(null)
      } else {
        setError("Por favor seleccione un archivo CSV válido")
      }
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch('http://159.65.106.247:8000/csv_classify', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Error uploading file:', error)
      setError(error instanceof Error ? error.message : 'Error desconocido al procesar el archivo')
    }

    setUploading(false)
  }

  const downloadResults = () => {
    if (!result?.data) return
    
    // Convert data array to CSV
    const headers = Object.keys(result.data[0])
    const csvContent = [
      headers.join(','),
      ...result.data.map(row => headers.map(header => JSON.stringify(row[header] || '')).join(','))
    ].join('\n')
    
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.style.display = 'none'
    a.href = url
    a.download = 'classification_results.csv'
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
  }

  const getCategoryColor = (category: string) => {
    switch (category) {
      case "cardiovascular":
        return "text-blue-600"
      case "hepatorenal":
        return "text-green-600"
      case "neurological":
        return "text-purple-600"
      case "oncological":
        return "text-red-600"
      default:
        return "text-gray-600"
    }
  }

  const getCategoryBadgeColor = (category: string) => {
    switch (category) {
      case "cardiovascular":
        return "bg-blue-100 text-blue-800"
      case "hepatorenal":
        return "bg-green-100 text-green-800"
      case "neurological":
        return "bg-purple-100 text-purple-800"
      case "oncological":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <Brain className="h-8 w-8 text-primary" />
                <h1 className="text-3xl font-bold text-foreground">Clasificación Masiva de Artículos</h1>
              </div>
              <p className="text-muted-foreground text-lg">
                Cargar archivo CSV para clasificación y evaluación de múltiples artículos
              </p>
            </div>
            <div className="flex gap-2">
              <Link href="/">
                <Button variant="outline" className="flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Clasificación Individual
                </Button>
              </Link>
              <Link href="/rendimiento">
                <Button variant="outline" className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Rendimiento del Modelo
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto space-y-8">
          
          {/* Instructions */}
          <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-blue-500" />
                Formato del Archivo CSV
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  El archivo CSV debe contener las siguientes columnas:
                </p>
                <div className="bg-muted/50 p-4 rounded-lg font-mono text-sm">
                  <div className="font-semibold mb-2">Columnas requeridas:</div>
                  <ul className="space-y-1">
                    <li><span className="text-blue-600">title</span> - Título del artículo</li>
                    <li><span className="text-green-600">abstract</span> - Resumen del artículo</li>
                    <li><span className="text-orange-600">group</span> - Categoría verdadera (cardiovascular, hepatorenal, neurological, oncological)</li>
                  </ul>
                </div>
                <p className="text-sm text-muted-foreground">
                  La salida incluirá una nueva columna <code>group_predicted</code> con las predicciones del modelo
                  y métricas de evaluación como F1-score ponderado, precisión, recall y matriz de confusión.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Upload Section */}
          <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5 text-primary" />
                Cargar Archivo CSV
              </CardTitle>
              <CardDescription>
                Seleccione o arrastre un archivo CSV para clasificar múltiples artículos
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Drag and Drop Area */}
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 cursor-pointer ${
                    dragActive
                      ? "border-primary bg-primary/5"
                      : "border-muted-foreground/25 hover:border-primary/50"
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById('file-upload')?.click()}
                >
                  <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <div className="space-y-2">
                    <p className="text-lg font-medium">
                      {file ? file.name : "Arrastre su archivo CSV aquí"}
                    </p>
                    <p className="text-muted-foreground">
                      o haga clic para seleccionar
                    </p>
                  </div>
                  <Input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                  />
                  <Button variant="outline" type="button" className="mt-4">
                    Seleccionar Archivo
                  </Button>
                </div>

                {/* File Info */}
                {file && (
                  <div className="flex items-center justify-between p-4 bg-muted/30 rounded-lg">
                    <div className="flex items-center gap-3">
                      <FileText className="h-5 w-5 text-primary" />
                      <div>
                        <p className="font-medium">{file.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {(file.size / 1024).toFixed(1)} KB
                        </p>
                      </div>
                    </div>
                    <Button
                      onClick={() => setFile(null)}
                      variant="ghost"
                      size="sm"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                )}

                {/* Error Message */}
                {error && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {/* Upload Button */}
                <Button
                  onClick={handleUpload}
                  disabled={!file || uploading}
                  className="w-full"
                  size="lg"
                >
                  {uploading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary-foreground border-t-transparent mr-2" />
                      Procesando archivo...
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4 mr-2" />
                      Clasificar y Evaluar
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Results Section */}
          {result && (
            <div className="space-y-6">
              {/* Overall Metrics */}
              {result.evaluation_metrics && (
                <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5 text-green-500" />
                      Métricas de Evaluación
                    </CardTitle>
                    <CardDescription>
                      Rendimiento del modelo en {result.total_processed} artículos procesados
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
                        <div className="text-2xl font-bold text-blue-600">
                          {(result.evaluation_metrics.f1_scores.weighted * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-blue-800 font-medium">F1-Score Ponderado</div>
                      </div>
                      <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
                        <div className="text-2xl font-bold text-green-600">
                          {(result.evaluation_metrics.accuracy.subset_accuracy * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-green-800 font-medium">Exactitud Subconjunto</div>
                      </div>
                      <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
                        <div className="text-2xl font-bold text-purple-600">
                          {(result.evaluation_metrics.accuracy.jaccard_similarity * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-purple-800 font-medium">Similitud Jaccard</div>
                      </div>
                      <div className="text-center p-4 bg-orange-50 rounded-lg border border-orange-200">
                        <div className="text-2xl font-bold text-orange-600">
                          {(result.evaluation_metrics.f1_scores.samples * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-orange-800 font-medium">F1-Score Muestras</div>
                      </div>
                    </div>
                    
                    {/* Additional Metrics */}
                    <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                      <div className="text-center p-3 bg-gray-50 rounded border">
                        <div className="font-semibold text-gray-700">
                          {(result.evaluation_metrics.hamming_loss * 100).toFixed(2)}%
                        </div>
                        <div className="text-gray-600">Pérdida Hamming</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded border">
                        <div className="font-semibold text-gray-700">
                          {(result.evaluation_metrics.f1_scores.macro * 100).toFixed(1)}%
                        </div>
                        <div className="text-gray-600">F1-Score Macro</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded border">
                        <div className="font-semibold text-gray-700">
                          {(result.evaluation_metrics.f1_scores.micro * 100).toFixed(1)}%
                        </div>
                        <div className="text-gray-600">F1-Score Micro</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Per-Class Metrics */}
              {result.evaluation_metrics && (
                <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle>F1-Score por Categoría</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left p-2">Categoría</th>
                            <th className="text-right p-2">F1-Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(result.evaluation_metrics.f1_scores.per_class).map(([category, f1Score]) => (
                            <tr key={category} className="border-b">
                              <td className={`p-2 font-medium capitalize ${getCategoryColor(category)}`}>
                                {category}
                              </td>
                              <td className="text-right p-2">{(f1Score * 100).toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Confusion Matrix */}
              {result.evaluation_metrics && (
                <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle>Matriz de Confusión</CardTitle>
                    <CardDescription>
                      Filas: Categorías reales | Columnas: Predicciones
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <div className="inline-block min-w-full">
                        <div className="text-sm">
                          <table className="border border-gray-300">
                            <thead>
                              <tr>
                                <th className="border border-gray-300 p-2 bg-gray-50"></th>
                                {result.evaluation_metrics.confusion_matrix.labels.map(category => (
                                  <th key={category} className="border border-gray-300 p-2 bg-gray-50 text-center font-medium capitalize">
                                    {category}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {result.evaluation_metrics.confusion_matrix.matrix.map((row: number[], i: number) => (
                                <tr key={i}>
                                  <th className="border border-gray-300 p-2 bg-gray-50 font-medium capitalize">
                                    {result.evaluation_metrics!.confusion_matrix.labels[i]}
                                  </th>
                                  {row.map((value, j) => (
                                    <td
                                      key={j}
                                      className={`border border-gray-300 p-2 text-center ${
                                        i === j ? 'bg-green-100 text-green-800 font-semibold' : value > 0 ? 'bg-red-50' : ''
                                      }`}
                                    >
                                      {value}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Individual Classifications */}
              <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-blue-500" />
                    Clasificaciones Individuales
                  </CardTitle>
                  <CardDescription>
                    Procesamiento secuencial de cada observación (primeras 50 filas)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="text-sm text-muted-foreground bg-blue-50 p-3 rounded-lg">
                      <strong>Proceso realizado:</strong> Cada fila fue procesada individualmente (título + resumen → clasificación → comparación con etiqueta real → cálculo de métricas globales)
                    </div>
                    
                    <div className="overflow-x-auto max-h-96 border rounded-lg">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-gray-50 border-b">
                          <tr>
                            <th className="text-left p-2">#</th>
                            <th className="text-left p-2 min-w-[200px]">Título</th>
                            <th className="text-center p-2">Real</th>
                            <th className="text-center p-2">Predicho</th>
                            <th className="text-center p-2">Resultado</th>
                            <th className="text-center p-2">Confianza</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.data.slice(0, 50).map((row, index) => {
                            const prediction = result.predictions[index];
                            const isCorrect = row.group === row.group_predicted;
                            const title = row.title?.substring(0, 60) + (row.title?.length > 60 ? '...' : '');
                            
                            return (
                              <tr key={index} className={`border-b hover:bg-gray-50 ${isCorrect ? 'bg-green-50/30' : 'bg-red-50/30'}`}>
                                <td className="p-2 font-mono text-xs">{index + 1}</td>
                                <td className="p-2">{title || 'Sin título'}</td>
                                <td className="p-2 text-center">
                                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCategoryBadgeColor(row.group)}`}>
                                    {row.group}
                                  </span>
                                </td>
                                <td className="p-2 text-center">
                                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCategoryBadgeColor(row.group_predicted)}`}>
                                    {row.group_predicted}
                                  </span>
                                </td>
                                <td className="p-2 text-center">
                                  {isCorrect ? (
                                    <CheckCircle className="w-4 h-4 text-green-600 mx-auto" />
                                  ) : (
                                    <XCircle className="w-4 h-4 text-red-600 mx-auto" />
                                  )}
                                </td>
                                <td className="p-2 text-center font-mono text-xs">
                                  {(prediction.confidence * 100).toFixed(1)}%
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    
                    {result.data.length > 50 && (
                      <p className="text-xs text-muted-foreground text-center bg-amber-50 p-2 rounded">
                        Mostrando las primeras 50 filas de {result.total_processed} procesadas. Descarga el CSV para ver todos los resultados.
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Download Button */}
              <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                <CardContent className="pt-6">
                  <Button onClick={downloadResults} className="w-full" size="lg">
                    <Download className="h-4 w-4 mr-2" />
                    Descargar Resultados CSV
                  </Button>
                  <p className="text-sm text-muted-foreground mt-2 text-center">
                    El archivo incluirá la columna 'group_predicted' con las clasificaciones del modelo
                  </p>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
