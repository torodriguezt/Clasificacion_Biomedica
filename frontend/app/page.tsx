"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import Link from "next/link"
import { Brain, FileText, Sparkles, BarChart3 } from "lucide-react"

type ClassificationResult = {
  categories: string[]
  highestCategory: string
  highestConfidence: number
}

export default function MedicalClassifier() {
  const [title, setTitle] = useState("")
  const [abstract, setAbstract] = useState("")
  const [result, setResult] = useState<ClassificationResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim() || !abstract.trim()) return

    setIsLoading(true)

    try {
      // Call the FastAPI backend
      const response = await fetch('http://localhost:8000/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: title.trim(),
          abstract: abstract.trim(),
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      // Mapeo de categorías del backend al frontend
      const categoryMapping: Record<string, string> = {
        'cardiovascular': 'Cardiovascular',
        'hepatorenal': 'Hepatorenal',
        'neurological': 'Neurológico',
        'oncological': 'Oncológico',
      }

      // Filtrar categorías que superen 0.36 (36%) de probabilidad - THRESHOLD REAL DEL MODELO
      const qualifyingCategories: string[] = []
      const threshold = 0.36

      Object.entries(data).forEach(([key, value]) => {
        if (key in categoryMapping && typeof value === 'number' && value > threshold) {
          qualifyingCategories.push(categoryMapping[key])
        }
      })

      // Si no hay categorías que superen el umbral, tomar la más alta
      if (qualifyingCategories.length === 0) {
        const dominantCategory = categoryMapping[data.dominant_category] || 'Cardiovascular'
        qualifyingCategories.push(dominantCategory)
      }

      const highestCategory = categoryMapping[data.dominant_category] || 'Cardiovascular'
      const highestConfidence = Math.round(data.confidence * 100)

      setResult({
        categories: qualifyingCategories,
        highestCategory,
        highestConfidence
      })
    } catch (error) {
      console.error('Error calling API:', error)
      
      // Fallback to simulated data if API fails
      const categories = ["Cardiovascular", "Hepatorenal", "Neurológico", "Oncológico"]
      const randomCategory = categories[Math.floor(Math.random() * categories.length)]
      const confidence = Math.floor(Math.random() * 30) + 70 // 70-99%

      setResult({
        categories: [randomCategory],
        highestCategory: randomCategory,
        highestConfidence: confidence
      })
      
      // Show error message to user
      alert('Error conectando con el backend. Mostrando datos simulados.')
    }

    setIsLoading(false)
  }

  const getCategoryColor = (category: string) => {
    switch (category) {
      case "Cardiovascular":
        return "bg-blue-500 text-white border-blue-600"
      case "Hepatorenal":
        return "bg-green-500 text-white border-green-600"
      case "Neurológico":
        return "bg-purple-500 text-white border-purple-600"
      case "Oncológico":
        return "bg-red-500 text-white border-red-600"
      default:
        return "bg-gray-500 text-white border-gray-600"
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
                <h1 className="text-3xl font-bold text-foreground">Clasificador de Artículos Biomédicos con IA</h1>
              </div>
              <p className="text-muted-foreground text-lg">
                Clasificación automática de literatura médica en 4 dominios especializados
              </p>
            </div>
            {/* Navigation Link */}
            <Link href="/rendimiento">
              <Button
                variant="outline"
                className="flex items-center gap-2 hover:bg-primary/10 transition-colors bg-transparent"
              >
                <BarChart3 className="h-4 w-4" />
                Rendimiento del Modelo
              </Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Classification Form */}
          <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary" />
                Datos del Artículo
              </CardTitle>
              <CardDescription>
                Ingrese el título y resumen del artículo biomédico para su clasificación
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="title" className="text-sm font-medium">
                    Título del artículo
                  </Label>
                  <Input
                    id="title"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="Ingrese el título del artículo biomédico..."
                    className="transition-all duration-200 focus:ring-2 focus:ring-primary/20"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="abstract" className="text-sm font-medium">
                    Resumen del artículo
                  </Label>
                  <Textarea
                    id="abstract"
                    value={abstract}
                    onChange={(e) => setAbstract(e.target.value)}
                    placeholder="Ingrese el resumen o abstract del artículo..."
                    className="min-h-32 transition-all duration-200 focus:ring-2 focus:ring-primary/20"
                    required
                  />
                </div>

                <Button
                  type="submit"
                  disabled={isLoading || !title.trim() || !abstract.trim()}
                  className="w-full bg-primary hover:bg-primary/90 text-primary-foreground transition-all duration-200 transform hover:scale-[1.02] disabled:transform-none"
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary-foreground border-t-transparent mr-2" />
                      Analizando con IA...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4 mr-2" />
                      Diagnosticar con IA
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Results Section */}
          {result && (
            <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm animate-fade-in">
              <CardHeader>
                <CardTitle className="text-xl">Resultado de la Clasificación</CardTitle>
                <CardDescription>
                  {result.categories.length === 1 
                    ? "El modelo identifica que el artículo pertenece a:" 
                    : "El modelo identifica que el artículo pertenece a múltiples categorías:"
                  }
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Categorías Identificadas */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Brain className="h-6 w-6 text-primary" />
                    <h3 className="text-lg font-semibold text-foreground">
                      {result.categories.length === 1 ? "Categoría:" : "Categorías:"}
                    </h3>
                  </div>
                  
                  <div className="flex flex-wrap gap-3">
                    {result.categories.map((category, index) => (
                      <span
                        key={index}
                        className={`px-6 py-3 rounded-full text-base font-semibold ${getCategoryColor(category)} transform transition-all duration-200 hover:scale-105`}
                      >
                        {category}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Información adicional si hay múltiples categorías */}
                {result.categories.length > 1 && (
                  <div className="p-4 rounded-lg bg-muted/30 border border-muted">
                    <p className="text-sm text-muted-foreground">
                      <strong>Nota:</strong> El artículo presenta características significativas de múltiples especialidades médicas. 
                      La categoría principal es <strong>{result.highestCategory}</strong> con {result.highestConfidence}% de confianza.
                    </p>
                  </div>
                )}

                {/* Información adicional si es una sola categoría */}
                {result.categories.length === 1 && (
                  <div className="p-4 rounded-lg bg-muted/30 border border-muted">
                    <p className="text-sm text-muted-foreground">
                      El modelo tiene <strong>{result.highestConfidence}% de confianza</strong> en esta clasificación.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Information Section */}
          <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-lg">¿Cómo funciona el modelo?</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground leading-relaxed">
                Nuestro modelo de inteligencia artificial utiliza técnicas avanzadas de procesamiento de lenguaje
                natural para analizar el contenido de artículos biomédicos y clasificarlos automáticamente en una de
                cuatro especialidades médicas principales.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                <div className="flex items-center gap-3 p-3 rounded-lg bg-blue-50 border border-blue-200">
                  <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                  <span className="text-sm font-medium text-blue-800">Cardiovascular</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-green-50 border border-green-200">
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                  <span className="text-sm font-medium text-green-800">Hepatorenal</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-purple-50 border border-purple-200">
                  <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                  <span className="text-sm font-medium text-purple-800">Neurológico</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-red-50 border border-red-200">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <span className="text-sm font-medium text-red-800">Oncológico</span>
                </div>
              </div>

              <p className="text-sm text-muted-foreground mt-4">
                La clasificación automática de literatura médica es fundamental para la organización eficiente del
                conocimiento científico, facilitando la investigación y el acceso a información relevante para
                profesionales de la salud.
              </p>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}