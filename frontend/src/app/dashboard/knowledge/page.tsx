"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Plus, Trash2, Upload } from "lucide-react";
import DashboardLayout from "@/components/layout/dashboard-layout";

interface KnowledgeBase {
  id: number;
  name: string;
  description: string;
  documents: Document[];
  created_at: string;
}

interface Document {
  id: number;
  title: string;
  content: string;
  file_path: string;
  created_at: string;
}

export default function KnowledgeBasePage() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);

  useEffect(() => {
    fetchKnowledgeBases();
  }, []);

  const fetchKnowledgeBases = async () => {
    try {
      const token = localStorage.getItem("token");
      const response = await fetch("http://localhost:8000/api/knowledge-base", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      const data = await response.json();
      setKnowledgeBases(data);
    } catch (error) {
      console.error("Failed to fetch knowledge bases:", error);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this knowledge base?"))
      return;

    try {
      const token = localStorage.getItem("token");
      const response = await fetch(
        `http://localhost:8000/api/knowledge-base/${id}`,
        {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        setKnowledgeBases((prev) => prev.filter((kb) => kb.id !== id));
      }
    } catch (error) {
      console.error("Failed to delete knowledge base:", error);
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">
              Knowledge Bases
            </h2>
            <p className="text-muted-foreground">
              Manage your knowledge bases and documents
            </p>
          </div>
          <Link
            href="/dashboard/knowledge/new"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Plus className="mr-2 h-4 w-4" />
            New Knowledge Base
          </Link>
        </div>

        <div className="grid gap-6">
          {knowledgeBases.map((kb) => (
            <div
              key={kb.id}
              className="rounded-lg border bg-card p-6 space-y-4"
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-lg font-semibold">{kb.name}</h3>
                  <p className="text-sm text-muted-foreground">
                    {kb.description || "No description"}
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    {kb.documents.length} documents •{" "}
                    {new Date(kb.created_at).toLocaleDateString()}
                  </p>
                </div>
                <div className="flex space-x-2">
                  <Link
                    href={`/dashboard/knowledge/${kb.id}/upload`}
                    className="inline-flex items-center justify-center rounded-md bg-secondary px-3 py-2 text-sm font-medium text-secondary-foreground hover:bg-secondary/80"
                  >
                    <Upload className="mr-2 h-4 w-4" />
                    Upload
                  </Link>
                  <button
                    onClick={() => handleDelete(kb.id)}
                    className="inline-flex items-center justify-center rounded-md bg-destructive px-3 py-2 text-sm font-medium text-destructive-foreground hover:bg-destructive/90"
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Delete
                  </button>
                </div>
              </div>

              {kb.documents.length > 0 && (
                <div className="border-t pt-4">
                  <h4 className="text-sm font-medium mb-2">Documents</h4>
                  <div className="space-y-2">
                    {kb.documents.map((doc) => (
                      <div
                        key={doc.id}
                        className="flex justify-between items-center rounded-md bg-accent/50 p-2 text-sm"
                      >
                        <span>{doc.title}</span>
                        <span className="text-muted-foreground">
                          {new Date(doc.created_at).toLocaleDateString()}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}

          {knowledgeBases.length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                No knowledge bases found. Create one to get started.
              </p>
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
