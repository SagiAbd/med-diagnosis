"use client";

import { useEffect, useState } from "react";
import DashboardLayout from "@/components/layout/dashboard-layout";
import { Book, MessageSquare, ArrowRight, Plus } from "lucide-react";
import { api, ApiError } from "@/lib/api";

interface KnowledgeBase {
  id: number;
  name: string;
  description: string;
  documents: any[];
}

interface Chat {
  id: number;
  title: string;
  messages: any[];
}

interface Stats {
  knowledgeBases: number;
  chats: number;
}

export default function DashboardPage() {
  const [stats, setStats] = useState<Stats>({ knowledgeBases: 0, chats: 0 });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [kbData, chatData] = await Promise.all([
          api.get("http://localhost:8000/api/knowledge-base"),
          api.get("http://localhost:8000/api/chat"),
        ]);

        setStats({
          knowledgeBases: kbData.length,
          chats: chatData.length,
        });
      } catch (error) {
        console.error("Failed to fetch stats:", error);
        if (error instanceof ApiError && error.status === 401) {
          return;
        }
      }
    };

    fetchStats();
  }, []);

  return (
    <DashboardLayout>
      <div className="space-y-8 p-8">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
              Welcome Back
            </h2>
            <p className="text-muted-foreground mt-2">
              Manage your knowledge bases and chat sessions
            </p>
          </div>
          <div className="flex gap-4">
            <a
              href="/dashboard/knowledge/new"
              className="inline-flex items-center justify-center rounded-lg bg-primary px-6 py-3 text-sm font-medium text-white hover:bg-primary/90 transition-colors shadow-lg shadow-primary/20"
            >
              <Plus className="mr-2 h-4 w-4" />
              New Knowledge Base
            </a>
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <div className="rounded-xl border bg-gradient-to-br from-primary/5 to-purple-500/5 p-8 backdrop-blur-sm transition-all hover:shadow-lg">
            <div className="flex items-center justify-between">
              <div className="space-y-4">
                <div className="rounded-full bg-primary/10 p-3 w-fit">
                  <Book className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">
                    Knowledge Bases
                  </p>
                  <h3 className="text-3xl font-bold mt-1">
                    {stats.knowledgeBases}
                  </h3>
                </div>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground" />
            </div>
          </div>

          <div className="rounded-xl border bg-gradient-to-br from-primary/5 to-purple-500/5 p-8 backdrop-blur-sm transition-all hover:shadow-lg">
            <div className="flex items-center justify-between">
              <div className="space-y-4">
                <div className="rounded-full bg-primary/10 p-3 w-fit">
                  <MessageSquare className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Chat Sessions</p>
                  <h3 className="text-3xl font-bold mt-1">{stats.chats}</h3>
                </div>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground" />
            </div>
          </div>
        </div>

        <div className="rounded-xl border bg-gradient-to-br from-primary/5 to-purple-500/5 p-8">
          <h3 className="text-xl font-semibold mb-6">Getting Started</h3>
          <div className="flex flex-col gap-4">
            <div className="flex items-start gap-4 rounded-lg border p-6 transition-all hover:border-primary hover:shadow-lg">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
                1
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Create Knowledge Base</h4>
                  <a href="/dashboard/knowledge/new">
                    <ArrowRight className="h-4 w-4 text-muted-foreground hover:text-primary" />
                  </a>
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Start by creating a new knowledge base and uploading your
                  documents
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4 rounded-lg border p-6 transition-all hover:border-primary hover:shadow-lg">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
                2
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Upload Documents</h4>
                  <a href="/dashboard/knowledge">
                    <ArrowRight className="h-4 w-4 text-muted-foreground hover:text-primary" />
                  </a>
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Upload your PDF, DOCX, MD or TXT files to build your knowledge
                  base
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4 rounded-lg border p-6 transition-all hover:border-primary hover:shadow-lg">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
                3
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Start Chatting</h4>
                  <a href="/dashboard/chat/new">
                    <ArrowRight className="h-4 w-4 text-muted-foreground hover:text-primary" />
                  </a>
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Chat with your knowledge base using AI to get instant answers
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
