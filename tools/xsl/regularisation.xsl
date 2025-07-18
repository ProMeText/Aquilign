<?xml version="1.0" encoding="UTF-8"?>
<!--Cette feuille régularise les éléments une fois tokénisés et xmlidsés-->
<!--Est-ce qu'on est obligé de produire du XML ? Ne peut-on pas produire une liste (id-forme-lemme) ?-->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:tei="http://www.tei-c.org/ns/1.0"
    exclude-result-prefixes="xs" version="2.0">

    <xsl:output method="xml"/>
    <xsl:strip-space elements="*"/>

    <xsl:param name="output_dir"/>


    <xsl:template match="@* | node()">
        <xsl:copy copy-namespaces="no">
            <xsl:apply-templates select="@* | node()"/>
        </xsl:copy>
    </xsl:template>



    <xsl:template match="/">
        <xsl:variable name="path">
            <xsl:value-of select="concat($output_dir, '?select=*tokenized.xml')"/>
        </xsl:variable>
        <xsl:for-each select="collection($path)//tei:TEI">
            <xsl:variable name="nom_fichier" select="@xml:id"/>
            <xsl:result-document href="{$output_dir}/{$nom_fichier}.regularise.xml">
                <xsl:element name="TEI" namespace="http://www.tei-c.org/ns/1.0">
                    <xsl:attribute name="xml:id" select="$nom_fichier"/>
                    <xsl:apply-templates/>
                </xsl:element>
            </xsl:result-document>
        </xsl:for-each>
    </xsl:template>

    <xsl:template match="tei:hi[not(@rend = 'lettre_attente')]">
        <xsl:apply-templates/>
    </xsl:template>


    <xsl:template match="tei:hi[@rend = 'lettre_attente']"/>

    <xsl:template match="tei:unclear[tei:w] | tei:damage[tei:w]">
        <xsl:copy-of select="tei:w"/>
    </xsl:template>

    <xsl:template match="tei:unclear[not(tei:w)] | tei:damage[not(tei:w)]">
        <xsl:value-of select="."/>
    </xsl:template>

    <xsl:template
        match="tei:figure | tei:w[not(text()) and descendant::tei:corr/not(descendant::text())]"/>
    <!--Revient à exclure les w vide-->

    <xsl:template match="tei:reg | tei:expan">
        <xsl:value-of select="."/>
    </xsl:template>

    <xsl:template match="tei:subst">
        <xsl:apply-templates select="tei:add"/>
    </xsl:template>


    <xsl:template match="tei:choice">
        <xsl:apply-templates select="tei:reg"/>
        <xsl:apply-templates select="tei:expan"/>
        <!--Des fois on a des sic sans correction: dans ce cas, appliquer
        les règles sur les sic, sinon la forme est supprimée.-->
        <xsl:choose>
            <xsl:when test="not(tei:corr) and tei:sic">
                <xsl:apply-templates select="tei:sic"/>
            </xsl:when>
            <xsl:otherwise>
                <xsl:value-of select="tei:corr"/>
            </xsl:otherwise>
        </xsl:choose>
        <!--Des fois on a des sic sans correction: dans ce cas, appliquer
        les règles sur les sic, sinon la forme est supprimée.-->

    </xsl:template>

    <xsl:template match="tei:sic | tei:seg">
        <xsl:apply-templates/>
    </xsl:template>

    <xsl:template match="tei:add[not(@type = 'commentaire')]">
        <xsl:apply-templates/>
    </xsl:template>

    <xsl:template
        match="tei:note | tei:fw | tei:del | tei:add[@type = 'commentaire'] | tei:handShift"/>

    <xsl:template match="tei:lb[parent::tei:w] | tei:cb[parent::tei:w] | tei:pb[parent::tei:w]"/>

    <!--
    <xsl:template match="text()">
        <xsl:variable name="v1" select="replace(., '⁊', 'e')"/>
        <xsl:value-of select="$v1"/>
    </xsl:template>-->


</xsl:stylesheet>
