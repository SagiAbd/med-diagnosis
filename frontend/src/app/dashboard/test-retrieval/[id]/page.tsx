"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { useToast } from "@/components/ui/use-toast";
import { api, ApiError } from "@/lib/api";
import DashboardLayout from "@/components/layout/dashboard-layout";
import { Search, ArrowRight, Sparkles } from "lucide-react";

interface KnowledgeBase {
  id: number;
  name: string;
  description: string;
}

interface DiagnosisItem {
  rank: number;
  diagnosis: string;
  icd10_code: string;
  explanation: string;
}

interface DiagnosisResponse {
  diagnoses: DiagnosisItem[];
}

export default function TestPage({ params }: { params: { id: string } }) {
  const [query, setQuery] = useState("");
  const [diagnoses, setDiagnoses] = useState<DiagnosisItem[]>([]);
  const [knowledgeBase, setKnowledgeBase] = useState<KnowledgeBase | null>(null);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    const fetchKnowledgeBase = async () => {
      try {
        const data = await api.get(`/api/knowledge-base/${params.id}`);
        setKnowledgeBase(data);
      } catch (error) {
        console.error("Failed to fetch knowledge base:", error);
        if (error instanceof ApiError) {
          toast({
            title: "Error",
            description: error.message,
            variant: "destructive",
          });
        }
      }
    };

    fetchKnowledgeBase();
  }, [params.id]);

  const handleTest = async () => {
    if (!query) {
      toast({
        title: "Введите текст",
        description: "Пожалуйста, опишите симптомы пациента",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const data: DiagnosisResponse = await api.post(
        "/api/knowledge-base/diagnose",
        {
          symptoms: query,
        }
      );

      setDiagnoses(data.diagnoses);
    } catch (error) {
      toast({
        title: "Ошибка",
        description: error instanceof Error ? error.message : "Неизвестная ошибка",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  if (!knowledgeBase) {
    return null;
  }

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gradient-to-b from-background to-background/50">
        <div className="max-w-6xl mx-auto py-12 px-6">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/60">
              AI-диагностика
            </h1>
            <p className="mt-4 text-lg text-muted-foreground">
              <span className="font-semibold text-foreground">
                {knowledgeBase.name}
              </span>
              {knowledgeBase.description && <span className="mx-2">•</span>}
              <span className="italic">{knowledgeBase.description}</span>
            </p>
          </div>

          <Card className="backdrop-blur-sm bg-card/50 border-primary/20">
            <CardContent className="p-8">
              <div className="flex gap-4">
                <div className="relative flex-1">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <Search className="h-5 w-5 text-muted-foreground" />
                  </div>
                  <Input
                    placeholder="Опишите симптомы пациента (возраст, жалобы, анамнез)..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="pl-12 h-14 text-lg bg-background/50 border-primary/20 focus:border-primary"
                    onKeyDown={(e) => e.key === "Enter" && handleTest()}
                    disabled={loading}
                  />
                  <Button
                    onClick={handleTest}
                    size="lg"
                    className="absolute right-0 top-0 h-14 px-8 bg-primary hover:bg-primary/90"
                    disabled={loading}
                  >
                    {loading ? (
                      <span className="flex items-center">
                        <Sparkles className="animate-spin mr-2 h-4 w-4" />
                        Анализ...
                      </span>
                    ) : (
                      <span className="flex items-center">
                        Определить диагноз
                        <ArrowRight className="ml-2 h-4 w-4" />
                      </span>
                    )}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {diagnoses.length > 0 && (
            <div className="mt-12 space-y-8">
              <h2 className="text-2xl font-semibold flex items-center gap-2">
                <Sparkles className="h-6 w-6 text-primary" />
                Возможные диагнозы
              </h2>
              <div className="grid gap-6">
                {diagnoses.map((item) => (
                  <Card
                    key={item.rank}
                    className="overflow-hidden border-0 shadow-lg hover:shadow-xl transition-shadow duration-300 bg-card/50 backdrop-blur-sm"
                  >
                    <CardContent className="p-8">
                      <div className="flex items-start gap-6">
                        <span className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/10 text-primary font-bold text-lg flex items-center justify-center">
                          {item.rank}
                        </span>
                        <div className="flex-1 space-y-3">
                          <div className="flex items-center gap-3 flex-wrap">
                            <span className="text-xl font-semibold">
                              {item.diagnosis}
                            </span>
                            <span className="px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-mono font-medium">
                              {item.icd10_code}
                            </span>
                          </div>
                          <p className="text-muted-foreground leading-relaxed">
                            {item.explanation}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
