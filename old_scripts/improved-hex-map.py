def create_hex_map_with_value(self,
                            df: pd.DataFrame,
                            valor_total: float,
                            polygon_points: Optional[List[Tuple[float, float]]] = None,
                            years: Optional[List[int]] = None,
                            output_path: Optional[Path] = None,
                            shapefile_path: Optional[Path] = None,
                            column_mapping: Optional[dict] = None) -> Optional[gpd.GeoDataFrame]:
    try:
        # Añadir logging para diagnóstico
        logger.info(f"Procesando DataFrame inicial: {len(df)} registros")
        
        # 1. Verificar datos antes del procesamiento
        df = df.copy()
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y %H:%M', errors='coerce')
        
        # Filtrar registros con fechas inválidas
        df = df.dropna(subset=['Fecha'])
        logger.info(f"Registros después de filtrar fechas inválidas: {len(df)}")
        
        # 2. Crear el GeoDataFrame base
        hex_gdf = self.create_hex_map(
            df,
            polygon_points=polygon_points,
            years=years,
            output_path=None,
            save_shapefile=False,
            shapefile_path=None,
            column_mapping=column_mapping
        )
        
        if hex_gdf is None:
            return None
            
        # 3. Mejorar el cálculo de valores
        total_horas = hex_gdf['time_diff'].sum()
        if total_horas <= 0:
            logger.error("Total de horas es 0 o negativo")
            return None
            
        valor_por_hora = valor_total / total_horas
        logger.info(f"Valor por hora: ${valor_por_hora:,.2f}")
        
        # Calcular valores en millones y preservar más decimales
        hex_gdf['val_millon'] = (hex_gdf['time_diff'] * valor_por_hora) / 1_000_000
        hex_gdf['val_millon'] = hex_gdf['val_millon'].round(4)  # Aumentar precisión decimal
        
        # Logging de estadísticas para diagnóstico
        logger.info("\nEstadísticas de valores:")
        logger.info(f"Hexágonos con actividad: {(hex_gdf['val_millon'] > 0).sum()}")
        logger.info(f"Rango de valores: ${hex_gdf['val_millon'].min():,.4f} - ${hex_gdf['val_millon'].max():,.4f}")
        logger.info(f"Percentiles de valores:")
        for p in [25, 50, 75, 90, 95, 99]:
            logger.info(f"Percentil {p}: ${hex_gdf['val_millon'].quantile(p/100):,.4f}")
        
        # 4. Guardar resultados
        if shapefile_path:
            gdf_to_save = hex_gdf.copy()
            gdf_to_save = gdf_to_save.rename(columns={
                'time_diff': 'horas',
                'val_millon': 'val_mill',
                'h3_index': 'h3_idx'
            })
            
            # Asegurar que todos los valores sean numéricos válidos
            gdf_to_save['horas'] = gdf_to_save['horas'].fillna(0)
            gdf_to_save['val_mill'] = gdf_to_save['val_mill'].fillna(0)
            
            columns_to_keep = ['geometry', 'horas', 'val_mill', 'h3_idx']
            gdf_to_save = gdf_to_save[columns_to_keep]
            
            shapefile_path.parent.mkdir(parents=True, exist_ok=True)
            gdf_to_save.to_file(shapefile_path)
            
            # Guardar CSV adicional con todos los datos para análisis
            csv_path = shapefile_path.with_suffix('.csv')
            gdf_to_save.drop(columns=['geometry']).to_csv(csv_path, index=False)
        
        # 5. Crear visualización
        if output_path:
            self._create_hex_value_map(hex_gdf, output_path)
            
            # Crear mapa adicional con escala logarítmica
            log_output_path = output_path.with_stem(output_path.stem + '_log')
            self._create_hex_value_map(hex_gdf, log_output_path, use_log_scale=True)
        
        return hex_gdf
        
    except Exception as e:
        logger.error(f"Error en create_hex_map_with_value: {e}")
        return None

def _create_hex_value_map(self, hex_gdf: gpd.GeoDataFrame, output_path: Path, use_log_scale: bool = False) -> None:
    """Create and save visualization with improved value handling."""
    try:
        logger.info(f"Creando visualización ({'logarítmica' if use_log_scale else 'linear'})")
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Preparar datos para visualización
        data = hex_gdf.copy()
        if use_log_scale:
            # Agregar pequeño valor para evitar log(0)
            min_non_zero = data[data['val_millon'] > 0]['val_millon'].min()
            data['val_millon'] = data['val_millon'].apply(lambda x: np.log10(x + min_non_zero/10) if x > 0 else 0)
        
        # Configurar esquema de colores
        cmap = plt.cm.YlOrRd
        cmap.set_under('lightgrey')
        
        # Calcular límites de visualización
        non_zero_values = data[data['val_millon'] > 0]['val_millon']
        if len(non_zero_values) > 0:
            vmin = non_zero_values.min()
            vmax = non_zero_values.quantile(0.99)  # Usar percentil 99 en lugar de 95
        else:
            vmin = 0
            vmax = 1
        
        # Plotear hexágonos
        data.plot(
            column='val_millon',
            cmap=cmap,
            legend=True,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                'label': 'Valor (Millones $)' if not use_log_scale else 'Log10(Valor)',
                'orientation': 'vertical',
                'extend': 'both'
            }
        )
        
        # Agregar mapa base
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            zoom=8
        )
        
        # Personalizar plot
        scale_type = "logarítmica" if use_log_scale else "linear"
        plt.title(f'Mapa de valor por hexágono (Escala {scale_type})', pad=20, size=14)
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        
        # Guardar mapa
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Logging de estadísticas
        logger.info(f"\nEstadísticas del mapa de valor ({scale_type}):")
        logger.info(f"Rango de visualización: {vmin:.4f} - {vmax:.4f}")
        
    except Exception as e:
        logger.error(f"Error creating value map: {e}")
        if plt.get_fignums():
            plt.close()
