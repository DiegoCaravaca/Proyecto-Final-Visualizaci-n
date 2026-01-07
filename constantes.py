DTYPES = {
    "id": "int64",
    "store_nbr": "int64",
    "family": "category",
    "sales": "float64",
    "onpromotion": "int64",
    "holiday_type": "category",
    "locale": "category",
    "locale_name": "category",
    "description": "category",
    "transferred": "category",
    "dcoilwtico": "float64",
    "city": "category",
    "state": "category",
    "store_type": "category",
    "cluster": "int64",
    "transactions": "float64",
    "year": "int64",
    "month": "int64",
    "week": "UInt32",
    "quarter": "int64",
    "day_of_week": "category"
}

# -----------------------------------------------------------------------------
# 1. DICCIONARIO DE CIUDADES (Coordenadas Exactas de tu lista)
# -----------------------------------------------------------------------------
COORDENADAS_CIUDADES = {
    "Quito":         (-0.1807, -78.4678),
    "Cayambe":       (0.0414, -78.1552),
    "Latacunga":     (-0.9393, -78.6156),
    "Riobamba":      (-1.6650, -78.6483),
    "Ibarra":        (0.3517, -78.1223),
    "Santo Domingo": (-0.2530, -79.1754),
    "Guaranda":      (-1.5926, -79.0010),
    "Puyo":          (-1.4924, -77.9991),
    "Ambato":        (-1.2491, -78.6168),
    "Guayaquil":     (-2.1709, -79.9223),
    "Salinas":       (-2.2145, -80.9515),
    "Daule":         (-1.8616, -79.9777),
    "Babahoyo":      (-1.8022, -79.5344),
    "Quevedo":       (-1.0286, -79.4633),
    "Playas":        (-2.6319, -80.3881),
    "Libertad":      (-2.2312, -80.9006),
    "Cuenca":        (-2.9001, -79.0059),
    "Loja":          (-3.9931, -79.2042),
    "Machala":       (-3.2581, -79.9551),
    "Esmeraldas":    (0.9682, -79.6517),
    "Manta":         (-0.9621, -80.7127),
    "El Carmen":     (-0.2720, -79.4646),
    
    # Valor por defecto (Centro de Ecuador) por si apareciera alguna nueva
    "DEFAULT":       (-1.8312, -78.1834)
}